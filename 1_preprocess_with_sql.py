import os
import re
import json
import logging
import spacy
import db

WORKING_DIR = "./source_data/"


def configure_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def detect_speaker(sent_text):
    m = re.search(r"\b(?:said|replied|asked|whispered|shouted)\s+([A-Z][a-zA-Z]+)\b", sent_text)
    if m:
        return m.group(1)

    return None


def main():
    configure_logging()
    os.makedirs(WORKING_DIR, exist_ok=True)
    db_path = os.path.join(WORKING_DIR, "character_graph.db")
    db.init_db(db_path)
    conn = db.get_conn(db_path)

    nlp = spacy.load("en_core_web_sm")
    with open("book.txt", "r", encoding="utf-8") as f:
        text = f.read()

    doc = nlp(text)
    sentences = list(doc.sents)

    relationships = []
    dialogues = []

    for sent in sentences:
        sent_text = sent.text.strip()
        persons_orgs = [ent.text for ent in sent.ents if ent.label_ in ("PERSON", "ORG")]
        if len(persons_orgs) >= 2:
            rel = {
                "entity1": persons_orgs[0],
                "entity2": persons_orgs[1],
                "relationship": "interacted",
                "context": sent_text,
            }
            relationships.append(rel)
            # write to DB
            try:
                db.insert_relationship(conn, rel['entity1'], rel['entity2'], rel['relationship'], rel['context'])
            except Exception as e:
                logging.warning("Failed to insert relationship: %s", e)

        # dialogue detection (very simple): quoted substrings or reporting verbs
        m = re.search(r'(["\']).*?\1', sent_text)
        speaker = None
        dialogue_text = None
        if m:
            dialogue_text = m.group(0).strip('"\'')
            speaker = detect_speaker(sent_text)
        else:
            speaker = detect_speaker(sent_text)
            if speaker and re.search(r"\b(?:said|asked|replied|whispered|shouted)\b", sent_text, re.IGNORECASE):
                dialogue_text = sent_text

        if dialogue_text or speaker:
            d = {"speaker": speaker, "dialogue": dialogue_text or "", "context": "conversation"}
            dialogues.append(d)
            try:
                db.insert_dialogue(conn, d['speaker'], d['dialogue'], d['context'])
            except Exception as e:
                logging.warning("Failed to insert dialogue: %s", e)

    # write JSON outputs for compatibility
    with open(os.path.join(WORKING_DIR, "entities_relationships.json"), "w", encoding="utf-8") as f:
        json.dump(relationships, f, ensure_ascii=False, indent=2)
    with open(os.path.join(WORKING_DIR, "entity_dialogues.json"), "w", encoding="utf-8") as f:
        json.dump(dialogues, f, ensure_ascii=False, indent=2)

    logging.info("Processed %d sentences, wrote %d relationships and %d dialogues", len(sentences), len(relationships), len(dialogues))


if __name__ == "__main__":
    main()
