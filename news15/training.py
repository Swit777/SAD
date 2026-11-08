"""Current English training: grouped data, matched input lengths and sealed tests."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import logit
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline

from .audit import ContaminationIndex, committee_fingerprint, title_key, write_json
from .consensus import AuditCleaner, CalibratedConsensus, raw_score, policy_statistics
from .data import clean_text, combine_text, file_sha256
from .model import load_bundle, save_bundle, evaluate_predictions, committee_for_input
from .paths import PROJECT_DIR, MODEL_PATH, REPORTS_DIR

SEED = 42
ROLE_WEIGHTS = [('train',.70),('tuning',.075),('calibration_members',.05),
                ('calibration_fusion',.05),('policy',.05),('internal_group_test',.075)]
STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS)-{'no','not','nor'})


def archive_texts(path, with_labels=False):
    """Before model freeze this function exposes text for exclusion only, no labels."""
    rows=[]
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith('.txt') or '/__MACOSX/' in name:
                continue
            segments=name.split('/')
            if len(segments)!=4 or segments[2] not in ('fake','legit'):
                continue
            text=archive.read(name).decode('utf-8-sig',errors='strict').strip()
            lines=[line.strip() for line in text.splitlines() if line.strip()]
            if not lines:continue
            stem=Path(name).stem
            pair=re.sub(r'[._-]?(?:fake|legit)$','',stem)
            dataset='FakeNewsAMT' if segments[1]=='fakeNewsDataset' else 'Celebrity'
            row={'id':name,'title':lines[0],'text':' '.join(lines[1:]),
                 'pair_id':dataset+'/'+pair,'dataset':dataset}
            if with_labels:row['label']=int(segments[2]=='fake')
            rows.append(row)
    return pd.DataFrame(rows)


def learning_partitions(frame):
    """Merge exact headline groups, quarantine contradictions, preserve all old reserves."""
    frame=frame.copy()
    parents={str(value):str(value) for value in frame.group_id}
    def find(value):
        while parents[value]!=value:
            parents[value]=parents[parents[value]];value=parents[value]
        return value
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parents[max(a,b)]=min(a,b)
    titles={}
    for row in frame.itertuples():
        key=title_key(row.title)
        if key:
            if key in titles:union(str(row.group_id),titles[key])
            else:titles[key]=str(row.group_id)
    frame['learning_group']=[find(str(g)) for g in frame.group_id]
    frame['origin']=np.where(frame.eligibility.eq('new_candidate_unassigned'),'WELFake','ISOT_overlap')
    group_rows=[]
    for key,part in frame.groupby('learning_group',sort=True):
        if part.label.nunique()!=1:role='quarantine_conflicting_labels'
        elif part.eligibility.str.startswith('reserved').any():role='prior_reserved'
        elif part.get('external_overlap',pd.Series(False,index=part.index)).any():role='reserved_new_external'
        elif not part.eligibility.isin(['legacy_train','new_candidate_unassigned']).all():role='ineligible'
        else:role='unassigned'
        group_rows.append({'learning_group':key,'label':int(part.label.iloc[0]),'role':role,
                           'origin':'ISOT_overlap' if part.origin.eq('ISOT_overlap').any() else 'WELFake'})
    groups=pd.DataFrame(group_rows).set_index('learning_group')
    remaining=groups.index[groups.role.eq('unassigned')].to_numpy()
    residual=1.
    for index,(role,fraction) in enumerate(ROLE_WEIGHTS):
        if index==len(ROLE_WEIGHTS)-1:chosen=remaining
        else:
            strata=groups.loc[remaining].label.astype(str)+'|'+groups.loc[remaining].origin
            if strata.value_counts().min()<2:strata=groups.loc[remaining].label
            chosen,remaining=train_test_split(remaining,train_size=fraction/residual,
                random_state=SEED+index,stratify=strata)
            residual-=fraction
        groups.loc[chosen,'role']=role
    frame['role']=frame.learning_group.map(groups.role)
    if frame.groupby('learning_group').role.nunique().max()!=1:
        raise RuntimeError('A news group crossed learning roles.')
    return frame


def learning_views(frame, mode):
    """One score per original article per evaluation view; no invented labels."""
    if mode=='headline':return frame.title.fillna('').astype(str).tolist()
    if mode=='short_prose':return [' '.join(combine_text(t,b).split()[:35]) for t,b in zip(frame.title,frame.text)]
    if mode=='short':
        result=[]
        for row in frame.itertuples():
            use_title=int(hashlib.sha256(str(row.id).encode()).hexdigest()[:2],16)%2==0
            result.append(str(row.title) if use_title else ' '.join(combine_text(row.title,row.text).split()[:35]))
        return result
    return [combine_text(t,b) for t,b in zip(frame.title,frame.text)]


def measured_metrics(y, probabilities, threshold, groups=None, repetitions=200):
    result=evaluate_predictions(y,probabilities,threshold)
    matrix=np.asarray(result['confusion_matrix'])
    result['false_fake_rate']=float(matrix[0,1]/max(1,matrix[0].sum()))
    result['real_recall']=float(matrix[0,0]/max(1,matrix[0].sum()))
    y=np.asarray(y,int);pred=np.asarray(probabilities)>=threshold
    ids=np.asarray(groups if groups is not None else np.arange(len(y))).astype(str)
    codes,unique=pd.factorize(ids)
    # Four confusion counts per group allow paired bootstrap without repeated text scoring.
    contributions=np.zeros((len(unique),4),dtype=int)
    np.add.at(contributions,(codes,y*2+pred.astype(int)),1)
    rng=np.random.default_rng(SEED);values=[];fprs=[]
    for _ in range(repetitions):
        a,b,c,d=contributions[rng.integers(0,len(unique),len(unique))].sum(axis=0)
        values.append(.5*(2*a/max(1,2*a+b+c)+2*d/max(1,2*d+b+c)))
        fprs.append(b/max(1,a+b))
    result['macro_f1_ci95']=np.quantile(values,[.025,.975]).tolist()
    result['false_fake_rate_ci95']=np.quantile(fprs,[.025,.975]).tolist()
    result['ci_method']='200 news-group bootstraps; paired true/fake stories resampled together when available.'
    return result


def committee_details(committee,texts,batch_size=128):
    batches=[committee.details(texts[i:i+batch_size]) for i in range(0,len(texts),batch_size)]
    return {key:np.concatenate([b[key] for b in batches]) for key in batches[0]}


def evaluate_frame(bundle,frame,mode='full'):
    texts=learning_views(frame,mode)
    if mode=='headline':committee=bundle.get('headline_consensus',bundle['consensus'])
    elif mode in ('short','short_prose'):committee=bundle.get('short_consensus',bundle.get('headline_consensus',bundle['consensus']))
    else:committee=bundle['consensus']
    details=committee_details(committee,texts)
    groups=frame['pair_id'] if 'pair_id' in frame else frame.get('learning_group',frame.id)
    metrics=measured_metrics(frame.label.to_numpy(int),details['probabilities'],committee.policy['threshold'],groups)
    unanimous=details['agreement']>=1-1e-9
    result={'metrics':metrics,'policy':policy_statistics(frame.label,details['probabilities'],details['member_probabilities'],committee.policy),
            'unanimous_count':int(unanimous.sum()),
            'unanimous_wrong_fraction':float((details['prediction'][unanimous]!=frame.label.to_numpy(int)[unanimous]).mean()) if unanimous.any() else None}
    return result,details


def choose_policy(y,probabilities,members):
    rows=[]
    for threshold in np.linspace(.25,.75,21):
        metric=evaluate_predictions(y,probabilities,float(threshold));matrix=np.asarray(metric['confusion_matrix'])
        fpr=float(matrix[0,1]/max(1,matrix[0].sum()))
        rows.append({'threshold':float(threshold),'f1_macro':metric['f1_macro'],'false_fake_rate':fpr})
    feasible=[r for r in rows if r['false_fake_rate']<=.20]
    best=max(feasible or rows,key=lambda r:(r['f1_macro'],-r['false_fake_rate'],-abs(r['threshold']-.5)))
    policies=[]
    for margin in [0,.05,.10,.15,.20,.25]:
        for consensus in [2/3,1.]:
            policy={'threshold':best['threshold'],'margin':margin,'consensus':consensus,'enabled':True}
            stats=policy_statistics(y,probabilities,members,policy)
            if stats['coverage']>=.5 and min(stats['automatic_fake_n'],stats['automatic_real_n'])>=30 and min(stats['fake_precision_wilson_lower'],stats['real_precision_wilson_lower'])>=.9:
                policies.append((stats['cost_per_article'],-stats['coverage'],policy))
    selected=min(policies,key=lambda item:item[:2])[2] if policies else {'threshold':best['threshold'],'margin':.15,'consensus':1.,'enabled':False}
    return selected,rows


def calibrate_committee(cleaner,vectorizer,estimators,matrices,parts):
    members=[]
    for name,estimator in estimators:
        score=raw_score(estimator,matrices['calibration_members'])
        calibrator=LogisticRegression(C=1.,max_iter=500)
        calibrator.fit(score.reshape(-1,1),parts['calibration_members'].label)
        members.append({'name':name,'estimator':estimator,'calibrator':calibrator})
    scores=np.column_stack([m['calibrator'].predict_proba(raw_score(m['estimator'],matrices['calibration_fusion']).reshape(-1,1))[:,1] for m in members])
    fusion=LogisticRegression(C=1.,max_iter=500).fit(logit(np.clip(scores.mean(axis=1),1e-6,1-1e-6)).reshape(-1,1),parts['calibration_fusion'].label)
    committee=CalibratedConsensus(cleaner,vectorizer,members,fusion,{'threshold':.5,'margin':.15,'consensus':1.,'enabled':False})
    detail=committee.details_from_matrix(matrices['policy'])
    committee.policy,grid=choose_policy(parts['policy'].label.to_numpy(int),detail['probabilities'],detail['member_probabilities'])
    return committee,grid


def train_project():
    started=datetime.now(timezone.utc).isoformat()
    protocol_path=PROJECT_DIR/'protocols/v2.json'
    plan=json.loads(protocol_path.read_text(encoding='utf-8'))['english_improvement']
    archive=PROJECT_DIR/plan['external_raw_archive']
    if file_sha256(archive)!=plan['external_archive_sha256']:raise ValueError('External archive checksum mismatch.')
    previous_report=REPORTS_DIR/'training_report.json'
    if previous_report.exists():
        previous=json.loads(previous_report.read_text(encoding='utf-8'))
        if previous.get('source',{}).get('sha256')==plan['external_archive_sha256']:
            raise ValueError('This external archive has already been evaluated. Keep the current model or preregister a new independent test before another training run.')
    work=Path(tempfile.gettempdir())/'news15-learning';work.mkdir(exist_ok=True)
    progress=lambda text:print(text,flush=True)
    progress('Reading eligible article labels and excluding sealed external texts without labels...')
    external_text=archive_texts(archive,False)
    exclusion=ContaminationIndex();exclusion.add(external_text)
    rows=[]
    for batch in pq.ParquetFile(PROJECT_DIR/'data/processed/corpus_v3/articles.parquet').iter_batches(batch_size=500):
        for row in batch.to_pylist():
            row['text']=str(row['text'] or '')[:6000]
            row['external_overlap']=exclusion.match(row['title'],row['text']) is not None
            rows.append({k:row[k] for k in ['id','group_id','title','text','label','published_at','eligibility','external_overlap']})
    frame=learning_partitions(pd.DataFrame(rows));del rows;gc.collect()
    groups=frame[frame.role.eq('train')].groupby('learning_group',sort=True).first()
    if len(groups)>plan['train_capacity']:
        chosen,_=train_test_split(groups.index,train_size=plan['train_capacity'],random_state=SEED,stratify=groups.label.astype(str)+'|'+groups.origin)
        frame.loc[frame.role.eq('train') & ~frame.learning_group.isin(chosen),'role']='unused_capacity'
    roles=frame[['id','group_id','learning_group','role','label','origin','eligibility']]
    roles.to_parquet(PROJECT_DIR/'data/processed/training_roles.parquet',index=False)
    parts={role:part.reset_index(drop=True) for role,part in frame.groupby('role') if role in dict(ROLE_WEIGHTS)}
    progress('Roles: '+json.dumps({r:len(p) for r,p in parts.items()}))
    cleaner=AuditCleaner(True,max_chars=plan['features']['maximum_cleaned_characters'])
    vectorizer=FeatureUnion([
        ('word',TfidfVectorizer(max_features=20000,ngram_range=(1,2),min_df=3,max_df=.98,sublinear_tf=True,stop_words=STOP_WORDS,dtype=np.float32)),
        ('char',TfidfVectorizer(max_features=6000,analyzer='char_wb',ngram_range=(3,4),min_df=5,sublinear_tf=True,dtype=np.float32)),
    ],transformer_weights={'word':1.,'char':.4},n_jobs=1)
    train=parts['train'];train_views=learning_views(train,'full')+learning_views(train,'short')
    progress('Fitting word and character TF-IDF on training views only...')
    X=vectorizer.fit_transform(cleaner.transform(train_views));del train_views;gc.collect()
    y=np.tile(train.label.to_numpy(int),2)
    weights=np.asarray([len(y)/(2*int((y==value).sum()))/2 for value in y],dtype=np.float32)
    tuning={mode:vectorizer.transform(cleaner.transform(learning_views(parts['tuning'],mode))) for mode in ['full','short']}
    background=np.asarray(X.mean(axis=0)).reshape(-1)
    best={};comparison=[]
    candidates=[('logistic_sgd',{'alpha':a},SGDClassifier(loss='log_loss',alpha=a,max_iter=25,tol=1e-3,n_iter_no_change=3,average=True,random_state=SEED)) for a in [1e-5,1e-4]]
    candidates += [('margin_sgd',{'alpha':a},SGDClassifier(loss='hinge',alpha=a,max_iter=25,tol=1e-3,n_iter_no_change=3,average=True,random_state=SEED)) for a in [1e-5,1e-4]]
    candidates += [('complement_nb',{'alpha':a},ComplementNB(alpha=a)) for a in [.5,1.]]
    for family,params,estimator in candidates:
        progress('Training '+family+' '+json.dumps(params))
        estimator.fit(X,y,sample_weight=weights)
        scores={mode:float(f1_score(parts['tuning'].label,raw_score(estimator,matrix)>=0,average='macro')) for mode,matrix in tuning.items()}
        score=float(np.mean(list(scores.values())))
        comparison.append({'model':family+' '+json.dumps(params),'family':family,'parameters':params,'validation_f1_macro':score,**scores})
        if family not in best or score>best[family][0]:best[family]=(score,estimator,params)
        progress('Tuning macro-F1 '+json.dumps(scores))
    del X,tuning;gc.collect()
    estimators=[(name,best[name][1]) for name in ['logistic_sgd','margin_sgd','complement_nb']]
    committees={};policy_grids={}
    for mode in ['full','short']:
        progress('Separate calibration and threshold selection: '+mode)
        matrices={role:vectorizer.transform(cleaner.transform(learning_views(parts[role],mode))) for role in ['calibration_members','calibration_fusion','policy']}
        committees[mode],policy_grids[mode]=calibrate_committee(cleaner,vectorizer,estimators,matrices,parts)
        del matrices;gc.collect()
    release={'status':'research_only','automatic_user_verdicts_enabled':False,'technical_gate_passed':False,
             'reason':'English statistical screening; factual verification and public deployment are not established.'}
    metadata={'dataset':'ISOT + WELFake, grouped article labels','language':'English','selected_model':'Calibrated word/character consensus: SGD logistic / SGD margin / ComplementNB',
        'training_method':'grouped_english','train_rows':len(train),'train_views':len(train)*2,'validation_rows':sum(len(parts[r]) for r in ['tuning','calibration_members','calibration_fusion','policy']),
        'test_rows':len(parts['internal_group_test']),'created_at_utc':started,'vocabulary_size':len(vectorizer.get_feature_names_out()),
        'selection_rule':'Mean full/short tuning macro-F1. Separate calibration roles and policy role. External labels opened only after freeze.',
        'release':release,'protocol_sha256':file_sha256(protocol_path),'baseline_model_sha256':file_sha256(MODEL_PATH),
        'limitations':['Scores identify corpus patterns, not independently verified truth.','English only.','Historical datasets do not establish reliability on current or personally written news.','Weak publisher and health-quality labels were excluded from training.']}
    primary=estimators[0][1]
    bundle={'schema_version':2,'pipeline':Pipeline([('clean',cleaner),('tfidf',vectorizer),('classifier',primary)]),
        'consensus':committees['full'],'headline_consensus':committees['short'],'background_mean':background,
        'metadata':metadata,'threshold':committees['full'].policy['threshold'],'consensus_policy':committees['full'].policy,
        'comparison':pd.DataFrame(comparison),'reference_scores':committees['full'].details(learning_views(parts['policy'],'full'))['probabilities']}
    fingerprints={m:committee_fingerprint(c) for m,c in committees.items()}
    save_bundle(bundle,work/'candidate.joblib')
    frozen={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'candidate_sha256':file_sha256(work/'candidate.joblib'),
            'fingerprints':fingerprints,'protocol_sha256':file_sha256(protocol_path),'external_labels_opened':False,
            'external_text_overlap_excluded':int(frame.external_overlap.sum())}
    write_json(work/'freeze.json',frozen)
    progress('Candidate and both decision policies frozen; evaluating held-out data...')
    baseline=load_bundle()
    evaluation={};baseline_evaluation={}
    for mode in ['full','headline']:
        evaluation['internal_'+mode],detail=evaluate_frame(bundle,parts['internal_group_test'],mode)
        if mode=='full':
            predictions=parts['internal_group_test'][['id','label','title']].copy()
            predictions['probability_fake']=detail['probabilities'];predictions['prediction']=detail['prediction']
            bundle['test_predictions']=predictions
    # Individual labels in this archive are accessed for the first time here.
    external=archive_texts(archive,True)
    for dataset,part in external.groupby('dataset'):
        for mode in ['full','headline']:
            key=dataset+'_'+mode
            evaluation[key],_=evaluate_frame(bundle,part,mode)
            baseline_evaluation[key],_=evaluate_frame(baseline,part,mode)
    old_ids=set(pd.read_csv(REPORTS_DIR/'audit_v2/after_external_predictions.csv',usecols=['id']).id.astype(str))
    raw=[]
    for source in ['politifact','gossipcop']:
        for kind,label in [('real',0),('fake',1)]:
            part=pd.read_csv(PROJECT_DIR/f'data/raw/fakenewsnet/{source}_{kind}.csv')
            part=part[part.id.astype(str).isin(old_ids)].copy();part['label']=label;part['text']=''
            raw.append(part[['id','title','text','label']])
    reused=pd.concat(raw).drop_duplicates('id').reset_index(drop=True)
    evaluation['FakeNewsNet_reused'],_=evaluate_frame(bundle,reused,'headline')
    baseline_evaluation['FakeNewsNet_reused'],_=evaluate_frame(baseline,reused,'headline')
    del baseline;gc.collect()
    if any(committee_fingerprint(committees[m])!=fingerprints[m] for m in committees):raise RuntimeError('External evaluation mutated frozen parameters.')
    progress('Running internal robustness and explanation checks...')
    checks=internal_checks(bundle,parts)
    final_metrics=evaluation['FakeNewsAMT_full']['metrics']
    metadata['test_metrics']=evaluation['internal_full']['metrics'];metadata['external_metrics']=final_metrics
    metadata['current_evaluation']=evaluation
    release['technical_gate_passed']=final_metrics['f1_macro']>=.7 and final_metrics['macro_f1_ci95'][0]>=.65
    report={'created_at_utc':started,'problem':'Reduce false accusations on real English news; preserve honest uncertainty.',
        'metadata':metadata,'comparison':comparison,'partitions':{r:{'n':len(p),'real':int(p.label.eq(0).sum()),'fake':int(p.label.eq(1).sum())} for r,p in parts.items()},
        'exclusions':frame.role.value_counts().to_dict(),'policy_grids':policy_grids,'policies':{m:c.policy for m,c in committees.items()},
        'evaluation':evaluation,'baseline_evaluation':baseline_evaluation,'freeze':frozen,'checks':checks,
        'external_status':{'FakeNewsAMT':'New to this project, paired human-written counterfactuals; labels opened after freeze.','Celebrity':'New to this project, paired entertainment news; labels opened after freeze.','FakeNewsNet':'Previously inspected headlines; diagnostic reuse, not independent confirmation.'},
        'source':{'url':plan['external_source'],'archive':str(archive.relative_to(PROJECT_DIR)),'sha256':file_sha256(archive),'news_records':len(external),
                  'limitations':'Author-provided dataset; local research only. Crowdsourced fabrications do not represent every possible manually written fake.'}}
    write_json(REPORTS_DIR/'training_report.json',report)
    save_bundle(bundle,work/'candidate.joblib')
    # Replace the one canonical model. No project model copies or version directories.
    shutil.copyfile(work/'candidate.joblib',MODEL_PATH)
    write_json(PROJECT_DIR/'artifacts/active_version.json',{'model_file':'artifacts/model.joblib','model_sha256':file_sha256(MODEL_PATH),
        'protocol_file':'protocols/v2.json','protocol_sha256':file_sha256(protocol_path),
        'report_file':'reports/training_report.json','report_sha256':file_sha256(REPORTS_DIR/'training_report.json'),
        'fingerprints':fingerprints,'baseline_model_sha256':metadata['baseline_model_sha256'],
        'source_files':{name:file_sha256(PROJECT_DIR/name) for name in ['news15/training.py','news15/model.py','news15/consensus.py','news15/audit.py','scripts/train.py','app.py']},
        'data_files':{p:file_sha256(PROJECT_DIR/p) for p in ['data/processed/corpus_v3/articles.parquet','data/processed/training_roles.parquet','data/raw/fakeNewsDatasets.zip']}})
    progress(json.dumps({'evaluation':{k:v['metrics'] for k,v in evaluation.items()},'baseline':{k:v['metrics'] for k,v in baseline_evaluation.items()}},ensure_ascii=True))
    return bundle


def internal_checks(bundle,parts):
    """Held-out robustness and primary-component attribution; no external-driven edits."""
    from sklearn.feature_selection import f_classif, SelectKBest
    from .experiments import _perturb
    sample=parts['internal_group_test'].sample(n=min(400,len(parts['internal_group_test'])),random_state=SEED)
    texts=learning_views(sample,'full');base=committee_details(bundle['consensus'],texts)
    results=[]
    for condition in ['original','uppercase','publisher_prefix','social_footer','typos','headline_only']:
        if condition=='original':detail=base
        elif condition=='headline_only':detail=committee_details(bundle['headline_consensus'],learning_views(sample,'headline'))
        else:
            if condition=='publisher_prefix':changed=['WASHINGTON (Reuters) - '+t for t in texts]
            elif condition=='social_footer':changed=[t+'\nFollow us: https://example.org/news' for t in texts]
            else:changed=[_perturb(t,'character_typos_8pct_words' if condition=='typos' else condition,SEED+i) for i,t in enumerate(texts)]
            detail=committee_details(bundle['consensus'],changed)
        results.append({'condition':condition,'n':len(sample),'macro_f1':float(f1_score(sample.label,detail['prediction'],average='macro')),
                        'class_flip_fraction':float((detail['prediction']!=base['prediction']).mean())})
    phi,base_value,logits=bundle['consensus'].component_shap(texts[:3],bundle['background_mean'])
    additivity=float(np.max(np.abs(phi.sum(axis=1)+base_value-logits)))
    train=parts['train'].sample(n=min(1600,len(parts['train'])),random_state=SEED)
    heldout=parts['tuning'].sample(n=min(600,len(parts['tuning'])),random_state=SEED)
    committee=bundle['consensus'];X=committee.vectorize(learning_views(train,'full'));T=committee.vectorize(learning_views(heldout,'full'))
    f_label=np.nan_to_num(f_classif(X,train.label)[0]);f_source=np.nan_to_num(f_classif(X,train.origin.eq('WELFake').astype(int))[0])
    overlap=len(set(np.argsort(f_label)[-100:]) & set(np.argsort(f_source)[-100:]))
    controls=[];rng=np.random.default_rng(SEED)
    for index in range(20):
        y=train.label.to_numpy(int) if index==0 else rng.permutation(train.label.to_numpy(int))
        selector=SelectKBest(f_classif,k=min(100,X.shape[1])).fit(X,y)
        model=LogisticRegression(C=1.,solver='liblinear',max_iter=200,random_state=SEED).fit(selector.transform(X),y)
        controls.append({'permutation':index,'labels_shuffled':index>0,'heldout_macro_f1':float(f1_score(heldout.label,model.predict(selector.transform(T)),average='macro'))})
    local=[]
    names=committee.vectorizer.get_feature_names_out()
    for i in range(3):
        order=np.argsort(np.abs(phi[i]))[-8:][::-1]
        local.append({'id':str(sample.iloc[i].id),'component_logit':float(logits[i]),'base':base_value,
                      'features':[{'feature':str(names[j]),'contribution':float(phi[i,j])} for j in order]})
    global_phi,_,_=committee.component_shap(texts[:120],bundle['background_mean'])
    importance=np.abs(global_phi).mean(axis=0)
    top=np.argsort(importance)[-30:][::-1]
    global_features=[{'feature':str(names[j]),'mean_absolute_component_SHAP':float(importance[j])} for j in top]
    deletions=[]
    for i,entry in enumerate(local):
        for feature in entry['features']:
            if not feature['feature'].startswith('word__'):continue
            term=feature['feature'].removeprefix('word__')
            changed=re.sub(r'\b'+re.escape(term)+r'\b',' ',texts[i],flags=re.I)
            after=float(committee.details([changed])['probabilities'][0])
            deletions.append({'id':entry['id'],'removed':term,'original_score':float(base['probabilities'][i]),'score_after_deletion':after})
    stability={}
    mode_keys=[('full','consensus'),('short_prose','short_consensus'),('headline','headline_consensus')] if bundle.get('short_consensus') is not None else [('full','consensus'),('short','headline_consensus')]
    for mode,key in mode_keys:
        policy_frame=parts['policy'];c=bundle[key]
        probability=committee_details(c,learning_views(policy_frame,mode))['probabilities']
        labels=policy_frame.label.to_numpy(int);group_codes,unique=pd.factorize(policy_frame.learning_group)
        refined=bundle.get('short_consensus') is not None
        threshold_grid=np.linspace(.25,.9,27) if refined else np.linspace(.25,.75,21)
        confusions=[]
        for threshold in threshold_grid:
            values=np.zeros((len(unique),4),dtype=int)
            np.add.at(values,(group_codes,labels*2+(probability>=threshold).astype(int)),1)
            confusions.append(values)
        rng=np.random.default_rng(SEED);chosen=[]
        for _ in range(60):
            sample_indices=rng.integers(0,len(unique),len(unique));scores=[];refined_rows=[]
            for threshold,values in zip(threshold_grid,confusions):
                a,b,fn,d=values[sample_indices].sum(axis=0)
                f1=.5*(2*a/max(1,2*a+b+fn)+2*d/max(1,2*d+b+fn));fpr=b/max(1,a+b)
                scores.append((fpr<=.20,f1,-fpr,-abs(threshold-.5),float(threshold)))
                refined_rows.append({'threshold':float(threshold),'f1_macro':f1,'false_fake_rate':fpr,'recall':d/max(1,d+fn),'classification_cost':(5*b+fn)/max(1,a+b+fn+d)})
            if refined:
                chosen.append(_refined_threshold_choice(refined_rows)['threshold'])
                continue
            feasible=[score for score in scores if score[0]]
            chosen.append(max(feasible or scores,key=lambda score:score[1:4])[4])
        stability[mode]={'repetitions':60,'threshold_ci95':np.quantile(chosen,[.025,.975]).tolist(),'frozen_threshold':c.policy['threshold'],'frozen_policy_replaced':False}
    return {'robustness':results,'component_shap':{'explains_final_consensus':False,'additivity_max_error':additivity,'background':'Training views only','local_examples':local},
            'circularity':{'top100_ANOVA_label_source_overlap':overlap,'source_proxy':'ISOT overlap versus WELFake-only; this differs from the historical Reuters proxy.',
                           'negative_controls':controls,'interpretation':'ANOVA and SHAP are diagnostics, never independent factual verification.'},
            'global_component_shap':global_features,'local_consensus_deletions':deletions,'boundary_stability':stability,
            'group_roles_disjoint':True,'weak_labels_excluded':True,'frozen_before_external_labels':True}


def _refined_threshold_choice(rows):
    optimum=max(r['f1_macro'] for r in rows)
    suitable=[r for r in rows if r['f1_macro']>=optimum-.005 and r['recall']>=.70]
    conservative=[r for r in suitable if r['false_fake_rate']<=.10] or [r for r in suitable if r['false_fake_rate']<=.20]
    return min(conservative or suitable or rows,key=lambda r:(r['classification_cost'],-r['f1_macro'],abs(r['threshold']-.5)))


def refined_policy(labels,probabilities,members):
    """Prefer fewer false accusations within a preregistered quality budget."""
    labels=np.asarray(labels,int);probabilities=np.asarray(probabilities,float)
    rows=[]
    for threshold in np.linspace(.25,.9,27):
        predicted=probabilities>=threshold
        tn=int(((labels==0)&~predicted).sum());fp=int(((labels==0)&predicted).sum())
        fn=int(((labels==1)&~predicted).sum());tp=int(((labels==1)&predicted).sum())
        score=.5*(2*tn/max(1,2*tn+fp+fn)+2*tp/max(1,2*tp+fp+fn))
        rows.append({'threshold':float(threshold),'f1_macro':float(score),'false_fake_rate':fp/max(1,tn+fp),
                     'recall':tp/max(1,tp+fn),'classification_cost':(5*fp+fn)/len(labels)})
    best=_refined_threshold_choice(rows)
    policies=[]
    for margin in [0,.05,.10,.15,.20,.25]:
        for consensus in [2/3,1.]:
            policy={'threshold':best['threshold'],'margin':margin,'consensus':consensus,'enabled':True}
            stats=policy_statistics(labels,probabilities,members,policy)
            if stats['coverage']>=.5 and min(stats['automatic_fake_n'],stats['automatic_real_n'])>=30 and min(stats['fake_precision_wilson_lower'],stats['real_precision_wilson_lower'])>=.9:
                policies.append((stats['cost_per_article'],-stats['coverage'],policy))
    policy=min(policies,key=lambda item:item[:2])[2] if policies else {'threshold':best['threshold'],'margin':.15,'consensus':1.,'enabled':False}
    return policy,rows


def refine_project():
    """Refine only calibration/fusion; all old external results are reused diagnostics."""
    from copy import copy
    from scipy.special import expit
    started=datetime.now(timezone.utc).isoformat()
    protocol_path=PROJECT_DIR/'protocols/v2.json'
    plan=json.loads(protocol_path.read_text(encoding='utf-8'))['refinement']
    if file_sha256(MODEL_PATH)!=plan['baseline_sha256']:
        raise ValueError('Current model differs from the preregistered refinement baseline. Do not repeat selection on scored external data.')
    roles_path=PROJECT_DIR/'data/processed/training_roles.parquet'
    if file_sha256(roles_path)!=plan['roles_sha256']:raise ValueError('Learning roles changed.')
    baseline=load_bundle();previous=json.loads((REPORTS_DIR/'training_report.json').read_text(encoding='utf-8'))
    baseline_hash=file_sha256(MODEL_PATH)
    progress=lambda message:print(message,flush=True)
    roles=pd.read_parquet(roles_path)
    needed=roles[roles.role.isin(['train','tuning','calibration_members','calibration_fusion','policy','internal_group_test','unused_capacity'])]
    frame=pd.read_parquet(PROJECT_DIR/'data/processed/corpus_v3/articles.parquet',
        columns=['id','title','text','label'],filters=[('id','in',needed.id.tolist())])
    frame['text']=frame.text.fillna('').str.slice(0,6000)
    frame=frame.merge(needed[['id','learning_group','role','origin']],on='id',validate='one_to_one')
    parts={r:p.reset_index(drop=True) for r,p in frame.groupby('role')}
    mapping={'full':'consensus','short_prose':'short_consensus','headline':'headline_consensus'}
    choices={};comparison=[];policy_grids={};tuning_checks=[]
    for mode,key in mapping.items():
        progress('Calibrating '+mode+' independently...')
        old=baseline['consensus'] if mode=='full' else baseline.get('short_consensus',baseline['headline_consensus'])
        if mode=='headline':old=baseline['headline_consensus']
        matrices={role:old.vectorize(learning_views(parts[role],mode)) for role in ['calibration_members','calibration_fusion','tuning','policy']}
        members=[]
        for member in old.members:
            calibrator=LogisticRegression(C=1.,max_iter=500).fit(raw_score(member['estimator'],matrices['calibration_members']).reshape(-1,1),parts['calibration_members'].label)
            members.append({**member,'calibrator':calibrator})
        member_scores={role:np.column_stack([m['calibrator'].predict_proba(raw_score(m['estimator'],matrix).reshape(-1,1))[:,1] for m in members]) for role,matrix in matrices.items() if role!='calibration_members'}
        candidates=[('current',old)]
        for fusion_mode in ['mean_logit','member_logits']:
            for c in [.01,.1,1.]:
                fusion=LogisticRegression(C=c,max_iter=500)
                committee=CalibratedConsensus(old.cleaner,old.vectorizer,members,fusion,
                    {'threshold':.5,'margin':.15,'consensus':1.,'enabled':False},fusion_mode=fusion_mode,
                    fusion_clip=.01 if fusion_mode=='member_logits' else 1e-6)
                fusion.fit(committee.fusion_features(member_scores['calibration_fusion']),parts['calibration_fusion'].label)
                candidates.append((fusion_mode+' C='+str(c),committee))
        scored=[]
        for name,committee in candidates:
            detail=committee.details_from_matrix(matrices['tuning'])
            metrics=evaluate_predictions(parts['tuning'].label,detail['probabilities'],.5)
            matrix=np.asarray(metrics['confusion_matrix']);fpr=float(matrix[0,1]/max(1,matrix[0].sum()))
            objective=metrics['f1_macro']-.05*fpr
            comparison.append({'model':mode+' / '+name,'format':mode,'family':name,'validation_f1_macro':metrics['f1_macro'],
                               'tuning_false_fake_rate':fpr,'selection_objective':objective})
            scored.append((objective,name=='current',name,committee))
        _,_,name,selected=max(scored,key=lambda item:item[:2])
        selected=copy(selected)
        detail=selected.details_from_matrix(matrices['policy'])
        selected.policy,policy_grids[mode]=refined_policy(parts['policy'].label.to_numpy(int),detail['probabilities'],detail['member_probabilities'])
        choices[mode]=selected
        old_tune=old.details_from_matrix(matrices['tuning']);new_tune=selected.details_from_matrix(matrices['tuning'])
        old_metrics=measured_metrics(parts['tuning'].label,old_tune['probabilities'],old.policy['threshold'],parts['tuning'].learning_group,repetitions=10)
        new_metrics=measured_metrics(parts['tuning'].label,new_tune['probabilities'],selected.policy['threshold'],parts['tuning'].learning_group,repetitions=10)
        candidate_metrics=new_metrics
        retained_current=new_metrics['f1_macro']<old_metrics['f1_macro']-.005 or new_metrics['false_fake_rate']>old_metrics['false_fake_rate']
        if retained_current:
            selected=copy(old);choices[mode]=selected;new_metrics=old_metrics
            name='current (quality protection)'
        tuning_checks.append({'format':mode,'selected':name,'before':old_metrics,'after':new_metrics,
                              'candidate_after':candidate_metrics,'retained_current':retained_current})
        progress(json.dumps({'format':mode,'selected':name,'threshold':selected.policy['threshold'],'tuning_before':old_metrics['f1_macro'],'tuning_after':new_metrics['f1_macro'],'false_fake_before':old_metrics['false_fake_rate'],'false_fake_after':new_metrics['false_fake_rate']}))
        del matrices;gc.collect()
    old_f1=np.mean([r['before']['f1_macro'] for r in tuning_checks]);new_f1=np.mean([r['after']['f1_macro'] for r in tuning_checks])
    old_fpr=np.mean([r['before']['false_fake_rate'] for r in tuning_checks]);new_fpr=np.mean([r['after']['false_fake_rate'] for r in tuning_checks])
    adoption=new_f1>=old_f1-.005 and new_fpr<=old_fpr
    if not adoption:
        write_json(REPORTS_DIR/'model_comparison.json',{'created_at_utc':started,'adopted':False,'comparison':comparison,'tuning':tuning_checks,
            'reason':'Preregistered tuning adoption criteria were not met. Canonical model left unchanged.'})
        progress('Refinement rejected on tuning criteria; current model retained.')
        return baseline
    bundle={**baseline,'metadata':deepcopy(baseline['metadata']),'comparison':pd.DataFrame(comparison)}
    for mode,key in mapping.items():bundle[key]=choices[mode]
    bundle['threshold']=choices['full'].policy['threshold'];bundle['consensus_policy']=choices['full'].policy
    bundle['pipeline']=Pipeline([('clean',choices['full'].cleaner),('tfidf',choices['full'].vectorizer),('classifier',choices['full'].members[0]['estimator'])])
    fingerprints={mode:committee_fingerprint(committee) for mode,committee in choices.items()}
    metadata=bundle['metadata']
    metadata.update({'created_at_utc':started,'selected_model':'Format-calibrated consensus with regularized stacking','input_modes':['headline','short','full'],
        'baseline_model_sha256':baseline_hash,'corpus_import_model_sha256':previous['metadata'].get('corpus_import_model_sha256',previous['metadata']['baseline_model_sha256']),
        'protocol_sha256':file_sha256(protocol_path),'selection_rule':'Frozen base estimators; format-specific calibration and regularized fusion selected on tuning. Thresholds selected on policy with a false-accusation cost; no external selection.'})
    freeze={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'fingerprints':fingerprints,'protocol_sha256':file_sha256(protocol_path),
            'external_labels_opened':False,'external_data_previously_seen':True,'base_estimators_retrained':False}
    progress('All three formats and thresholds frozen; evaluating confirmation and reused diagnostics...')
    evaluation={};baseline_evaluation={}
    for dataset,part in [('internal',parts['internal_group_test']),('unused_confirmation',parts['unused_capacity'])]:
        for mode in ['full','headline','short_prose']:
            key=dataset+'_'+('short' if mode=='short_prose' else mode)
            evaluation[key],detail=evaluate_frame(bundle,part,mode)
            baseline_evaluation[key],_=evaluate_frame(baseline,part,mode)
            if key=='internal_full':
                predictions=part[['id','title','label']].copy();predictions['probability_fake']=detail['probabilities'];predictions['prediction']=detail['prediction']
                bundle['test_predictions']=predictions
    external=archive_texts(PROJECT_DIR/'data/raw/fakeNewsDatasets.zip',True)
    for dataset,part in external.groupby('dataset'):
        for mode in ['full','headline']:
            key=dataset+'_'+mode;evaluation[key],_=evaluate_frame(bundle,part,mode);baseline_evaluation[key],_=evaluate_frame(baseline,part,mode)
    ids=set(pd.read_csv(REPORTS_DIR/'audit_v2/after_external_predictions.csv',usecols=['id']).id.astype(str))
    external_rows=[]
    for source in ['politifact','gossipcop']:
        for kind,label in [('real',0),('fake',1)]:
            part=pd.read_csv(PROJECT_DIR/f'data/raw/fakenewsnet/{source}_{kind}.csv')
            part=part[part.id.astype(str).isin(ids)].copy();part['label']=label;part['text']='';external_rows.append(part[['id','title','text','label']])
    external=pd.concat(external_rows).drop_duplicates('id').reset_index(drop=True)
    evaluation['FakeNewsNet_reused'],_=evaluate_frame(bundle,external,'headline')
    baseline_evaluation['FakeNewsNet_reused'],_=evaluate_frame(baseline,external,'headline')
    if fingerprints!={mode:committee_fingerprint(c) for mode,c in choices.items()}:raise RuntimeError('Diagnostic evaluation mutated parameters.')
    checks=internal_checks(bundle,parts)
    checks['external_data_previously_seen']=True
    metrics=evaluation['FakeNewsAMT_full']['metrics'];selective=evaluation['FakeNewsAMT_full']['policy']
    gate={'external_macro_f1':metrics['f1_macro']>=.7,'external_macro_f1_ci_lower':metrics['macro_f1_ci95'][0]>=.65,
          'external_automatic_fake_precision':selective['fake_precision_wilson_lower']>=.9,
          'external_automatic_real_precision':selective['real_precision_wilson_lower']>=.9,'external_coverage':selective['coverage']>=.5,
          'fresh_external_confirmation':False}
    metadata['release'].update({'checks':gate,'technical_gate_passed':all(gate.values()),'automatic_user_verdicts_enabled':False})
    metadata['test_metrics']=evaluation['internal_full']['metrics'];metadata['external_metrics']=metrics;metadata['current_evaluation']=evaluation
    report={**previous,'created_at_utc':started,'metadata':metadata,'comparison':comparison,'policies':{mode:c.policy for mode,c in choices.items()},
            'policy_grids':policy_grids,'evaluation':evaluation,'baseline_evaluation':baseline_evaluation,'freeze':freeze,'checks':checks,
            'refinement':{'base_estimators_retrained':False,'adopted':True,'tuning':tuning_checks,
                          'confirmation':'Previously unused within-corpus groups; not evidence of publisher transfer.'},
            'external_status':{name:'Previously inspected; reused diagnostic only. Not a fresh independent validation.' for name in ['FakeNewsAMT','Celebrity','FakeNewsNet']}}
    report['partitions']=dict(previous['partitions']);report['partitions']['unused_confirmation']={'n':len(parts['unused_capacity']),
        'real':int(parts['unused_capacity'].label.eq(0).sum()),'fake':int(parts['unused_capacity'].label.eq(1).sum())}
    bundle['reference_scores']=choices['full'].details(learning_views(parts['policy'],'full'))['probabilities']
    save_bundle(bundle,MODEL_PATH);write_json(REPORTS_DIR/'training_report.json',report)
    pointer=json.loads((PROJECT_DIR/'artifacts/active_version.json').read_text(encoding='utf-8'))
    pointer.update({'model_sha256':file_sha256(MODEL_PATH),'report_sha256':file_sha256(REPORTS_DIR/'training_report.json'),
                    'baseline_model_sha256':baseline_hash,'corpus_import_model_sha256':metadata['corpus_import_model_sha256'],
                    'fingerprints':fingerprints,'protocol_sha256':file_sha256(protocol_path)})
    for name in pointer.get('source_files',{}):pointer['source_files'][name]=file_sha256(PROJECT_DIR/name)
    write_json(PROJECT_DIR/'artifacts/active_version.json',pointer)
    progress(json.dumps({name:{'before':baseline_evaluation[name]['metrics']['f1_macro'],'after':entry['metrics']['f1_macro'],
        'false_fake_before':baseline_evaluation[name]['metrics']['false_fake_rate'],'false_fake_after':entry['metrics']['false_fake_rate']} for name,entry in evaluation.items()}))
    return bundle
