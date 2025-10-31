import os
import spacy
import sqlite3
import re
import logging
import shutil
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer

WORKING_DIR = "./source_data/"

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )

def call_ollama(prompt, model="phi3:mini", host="http://localhost:11434"):
    try:
        payload = {"model": model, "prompt": prompt}
        resp = requests.post(f"{host}/api/generate", json=payload, timeout=30)
        resp.raise_for_status()
        # Try to parse JSON response body; many Ollama setups return plain text or streaming NDJSON.
        text = resp.text.strip()
        # If response is JSON array/object, return parsed; otherwise return raw text
        try:
            return json.loads(text)
        except Exception:
            return text
    except Exception as e:
        logging.warning("Ollama call failed: %s", e)
        return None
    
class DataPreprocessor:

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.conn = sqlite3.connect(os.path.join(WORKING_DIR, "character_graph.db"))
        
    def preprocess_book(self):
        # Step 1: Split into tokens and sentences
        with open("./book.txt", "r", encoding="utf-8") as f:
            doc = self.nlp(f.read())
        
        # prepare batches once and reuse
        batches = self.prepare_batches(doc, max_tokens=1200)
        
        # Step 2: Identify entities and relationships (use precomputed sentence batches)
        entities_relationships = self.extract_entities_relationships(batches['sents'])
        
        # Step 3: Identify entities and dialogues (use precomputed quote batches) 
        entity_dialogues = self.extract_entity_dialogues(batches['quotes'])
        
        # Store in databases
        self.store_entities_relationships(entities_relationships)
        self.store_entity_dialogues(entity_dialogues)
        logging.info("preprocess_book() finished")

    def prepare_batches(self, doc, max_tokens=1200):
        """Return precomputed batches for sentences and quoted sentences.
        Result: {'sents': [...], 'quotes': [...]}
        Each element is a list-of-spacy-span batches (sum tokens <= max_tokens)."""
        sents = list(doc.sents)
        sents_batches = self.batch_spans_by_tokens(sents, max_tokens=max_tokens)
        quote_sents = [s for s in sents if ('"' in s.text or "'" in s.text)]
        quote_batches = self.batch_spans_by_tokens(quote_sents, max_tokens=max_tokens)
        return {'sents': sents_batches, 'quotes': quote_batches}

    #batch 
    def batch_spans_by_tokens(self, spans, max_tokens=1200):
        batches = []
        cur = []
        cur_tokens = 0
        for span in spans:
            tok_count = len(span)  # spaCy token count approximation
            if cur and cur_tokens + tok_count > max_tokens:
                batches.append(cur)
                cur = []
                cur_tokens = 0
            cur.append(span)
            cur_tokens += tok_count
        if cur:
            batches.append(cur)
        return batches
    
    def extract_entities_relationships(self, batches):
        entities = {}
        relationships = []
        
        for batch in batches:
            number = []
            for i, s in enumerate(batch):
                number.append(f"### SENTENCE {i}\n{s.text.strip()}")
            prompt = (
                "For each numbered sentence below extract PERSON and ORG entities and any relationships.\n"
                "Return a JSON array where each element is {\"sent_index\": int, \"relationships\": [ {\"entity1\":..., \"entity2\":..., \"relationship\":..., \"context\":...}, ... ] }\n\n"
                + "\n\n".join(number)
            )
            #1 using LLM
            llm_out = call_ollama(prompt)
            if isinstance(llm_out, list):
                # expecting a list where each element may be { "sent_index": int, "relationships": [...] }
                for item in llm_out:
                    sent_index = item.get("sent_index")
                    relations = item.get("relationships", [])
                    for obj in relations:
                        # basic validation
                        if all(k in obj for k in ("entity1", "entity2", "relationship")):
                            context = obj.get("context")
                            if not context and sent_index is not None and 0 <= sent_index < len(batch):
                                context = batch[sent_index].text
                            relationships.append({
                                "entity1": obj["entity1"],
                                "entity2": obj["entity2"],
                                "relationship": obj.get("relationship", "interacted"),
                                "context": context or ""
                            })
                continue  # skip to next sentence if LLM extraction succeeded
            
            #2 (if fallback use heuristic approach)
            for sent in batch:
                sent_entities = [ent.text for ent in sent.ents if ent.label_ in ['PERSON', 'ORG']]
                if len(sent_entities) >= 2:
                    relationships.append({
                        'entity1': sent_entities[0],
                        'entity2': sent_entities[1], 
                        'relationship': 'interacted',
                        'context': sent.text
                    })
        logging.info("Extracted entities and relationships:")
        for rel in relationships:
            logging.info(f"ENTITY1: {rel['entity1']}, ENTITY2: {rel['entity2']}, RELATIONSHIP: {rel['relationship']}")
        return {'entities': entities, 'relationships': relationships}
    
    def extract_entity_dialogues(self, batches):
        dialogues = []
        
        for batch in batches:
            number = []
            for i, s in enumerate(batch):
                number.append(f"### SENTENCE {i}\n{s.text.strip()}")
            prompt = (
                "For each numbered sentence below return JSON array of objects {\"sent_index\": int, \"speaker\": \"Name\" or null, \"dialogue\": \"quoted text\"}.\n\n"
                + "\n\n".join(number)
            )
            llm_out = call_ollama(prompt, timeout=120) if 'timeout' in call_ollama.__code__.co_varnames else call_ollama(prompt)
            if isinstance(llm_out, list):
                for item in llm_out:
                    speaker = item.get("speaker")
                    dialogue_text = item.get("dialogue")
                    if speaker:
                        dialogues.append({
                            'speaker': speaker,
                            'dialogue': dialogue_text,
                            'context': 'conversation'
                        })
                continue  # skip to next batch if LLM extraction succeeded
            else:
                # fallback to existing heuristic detection
                for sent in batch:
                    speaker = self.detect_speaker(sent)
                    if speaker:
                        m = re.search(r'(["\']).*?\1', sent.text)
                        dialogue_text = m.group(0).strip('"\'') if m else sent.text
                        dialogues.append({
                            'speaker': speaker,
                            'dialogue': dialogue_text,
                            'context': 'conversation'
                        })

        logging.info("Extracted entity dialogues:")
        for dialogue in dialogues:
            logging.info(f"SPEAKER: {dialogue['speaker']}, DIALOGUE: {dialogue['dialogue']}")
        return dialogues

    def detect_speaker(self, sent):
        """
        Heuristic speaker detection:
        - look for verbs like 'said', 'replied', 'asked' followed by a capitalized name
        - fallback to PERSON entities in the same sentence
        - fallback to PERSON entities in the previous sentence
        """
        text = sent.text
        # 1) regex-based common reporting verbs
        m = re.search(r'\b(?:said|replied|asked|whispered|shouted)\s+([A-Z][a-zA-Z]+)\b', text)
        if m:
            return m.group(1)

        # 2) try PERSON entities in this sentence
        persons = [ent.text for ent in sent.ents if ent.label_ == 'PERSON']
        if persons:
            return persons[0]

        # 3) fallback to previous sentence PERSON entity
        try:
            sents = list(sent.doc.sents)
            idx = sents.index(sent)
            if idx > 0:
                prev = sents[idx - 1]
                prev_persons = [ent.text for ent in prev.ents if ent.label_ == 'PERSON']
                if prev_persons:
                    return prev_persons[-1]
        except Exception:
            pass

        return None
    
    def store_entities_relationships(self, data):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS entities_relationships
                     (entity1 TEXT, entity2 TEXT, relationship TEXT, context TEXT)''')
        for rel in data['relationships']:
            c.execute('INSERT INTO entities_relationships VALUES (?,?,?,?)',
                      (rel['entity1'], rel['entity2'], rel['relationship'], rel['context']))
        self.conn.commit()
        logging.info("Stored entities and relationships in database")

        try:
            os.makedirs(WORKING_DIR, exist_ok=True)
            with open(os.path.join(WORKING_DIR, "entities_relationships.json"), "w", encoding="utf-8") as f:
                json.dump(data['relationships'], f, ensure_ascii=False, indent=2)
            logging.info("Wrote entities_relationships.json")
        except Exception as e:
            logging.error(f"Error writing entities_relationships.json: {e}")

    def store_entity_dialogues(self, dialogues):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS entity_dialogues
                     (speaker TEXT, dialogue TEXT, context TEXT)''')
        for dialogue in dialogues:
            c.execute('INSERT INTO entity_dialogues VALUES (?,?,?)',
                      (dialogue['speaker'], dialogue['dialogue'], dialogue['context']))
        self.conn.commit()
        logging.info("Stored entity dialogues in database")
        try:
            os.makedirs(WORKING_DIR, exist_ok=True)
            with open(os.path.join(WORKING_DIR, "entity_dialogues.json"), "w", encoding="utf-8") as f:
                json.dump(dialogues, f, ensure_ascii=False, indent=2)
            logging.info("Wrote entity_dialogues.json")
        except Exception as e:
            logging.error(f"Error writing entity_dialogues.json: {e}")

def main():

    # Always start with a fresh working directory: remove and recreate
    try:
        if os.path.exists(WORKING_DIR):
            shutil.rmtree(WORKING_DIR)
        os.makedirs(WORKING_DIR, exist_ok=True)
        logging.info(f"Cleaned and recreated working directory: {WORKING_DIR}")
    except Exception as e:
        logging.warning(f"Could not clean working directory {WORKING_DIR}: {e}")

    preprocessor = DataPreprocessor()
    preprocessor.preprocess_book()
    logging.info("main() exit")

if __name__ == "__main__":
    configure_logging()
    main()
    logging.info("main() exit")