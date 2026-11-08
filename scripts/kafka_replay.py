"""Optional localhost Kafka demonstration using an archived ISOT stream."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from news15.data import load_articles
from news15.model import load_bundle, predict_article


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["producer", "consumer"])
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", default="sad-news15")
    parser.add_argument("--max-events", type=int, default=200)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--from-beginning", action="store_true")
    args = parser.parse_args()
    if args.max_events < 1 or args.interval < 0:
        parser.error("--max-events must be positive; --interval cannot be negative")
    try:
        from kafka import KafkaConsumer, KafkaProducer
        from kafka.errors import KafkaError
    except ImportError:
        parser.exit(2, "Install the optional client first: python -m pip install kafka-python\n")

    try:
        if args.mode == "producer":
            articles = load_articles()
            rows = articles.loc[articles["split"].eq("test")].head(args.max_events)
            if rows.empty:
                parser.exit(2, "The prepared data contains no test articles.\n")
            producer = KafkaProducer(
                bootstrap_servers=args.bootstrap_servers.split(","),
                key_serializer=lambda value: value.encode("utf-8"),
                value_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
                acks="all",
            )
            try:
                for _, row in rows.iterrows():
                    event = {
                        "event_id": str(row["id"]),
                        "title": str(row["title"]),
                        "text": str(row["text"]),
                        "published_at": str(row["published_at"]),
                        "stream_kind": "archived_isot_replay",
                    }
                    producer.send(args.topic, key=event["event_id"], value=event).get(timeout=30)
                    print(json.dumps({"sent": event["event_id"]}), flush=True)
                    time.sleep(args.interval)
                producer.flush(timeout=30)
            finally:
                producer.close(timeout=10)
        else:
            bundle = load_bundle()
            consumer = KafkaConsumer(
                args.topic,
                bootstrap_servers=args.bootstrap_servers.split(","),
                group_id="sad-news15-demo",
                auto_offset_reset="earliest" if args.from_beginning else "latest",
                enable_auto_commit=False,
                consumer_timeout_ms=60_000,
                value_deserializer=lambda value: json.loads(value.decode("utf-8")),
            )
            count = 0
            try:
                for message in consumer:
                    event = message.value
                    result = predict_article(bundle, event.get("title", ""), event.get("text", ""))
                    output = {"event_id": event.get("event_id"), **result}
                    print(json.dumps(output, ensure_ascii=False, default=str), flush=True)
                    consumer.commit()
                    count += 1
                    if count >= args.max_events:
                        break
            finally:
                consumer.close()
            print(json.dumps({"processed": count}), flush=True)
    except FileNotFoundError as error:
        parser.exit(2, f"{error}\nPrepare data and train the model first; see README.md.\n")
    except KafkaError as error:
        parser.exit(2, f"Kafka is unavailable at {args.bootstrap_servers}: {error}\n")
    except KeyboardInterrupt:
        print("Stopped.", flush=True)


if __name__ == "__main__":
    main()
