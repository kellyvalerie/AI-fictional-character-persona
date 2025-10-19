import spacy
import sqlite3
import re
from sklearn.feature_extraction.text import TfidfVectorizer

class DataPreprocessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.conn = sqlite3.connect('character_graph.db')
        
    def preprocess_book(self, book_text):
        # Step 1: Split into tokens and sentences
        doc = self.nlp(book_text)
        
        # Step 2: Identify entities and relationships
        entities_relationships = self.extract_entities_relationships(doc)
        
        # Step 3: Identify entities and dialogues  
        entity_dialogues = self.extract_entity_dialogues(doc)
        
        # Store in databases
        self.store_entities_relationships(entities_relationships)
        self.store_entity_dialogues(entity_dialogues)
        print("preprocess_book() finished", flush=True)

    def extract_entities_relationships(self, doc):
        entities = {}
        relationships = []
        
        for sent in doc.sents:
            sent_entities = [ent.text for ent in sent.ents if ent.label_ in ['PERSON', 'ORG']]
            # Simple relationship extraction (you can enhance this)
            if len(sent_entities) >= 2:
                relationships.append({
                    'entity1': sent_entities[0],
                    'entity2': sent_entities[1], 
                    'relationship': 'interacted',  # Basic - enhance with ML
                    'context': sent.text
                })
        print("Extracted entities and relationships:")
        for rel in relationships:
            print(f"  ENTITY1: {rel['entity1']}, ENTITY2: {rel['entity2']}, RELATIONSHIP: {rel['relationship']}")
        return {'entities': entities, 'relationships': relationships}
    
    def extract_entity_dialogues(self, doc):
        dialogues = []
        # current_speaker = None
        # current_dialogue = []
        
        for sent in doc.sents:
            # Simple dialogue detection (enhance with quote patterns)
            if '"' in sent.text or "'" in sent.text:
                # Extract speaker from previous context
                speaker = self.detect_speaker(sent)
                if speaker:
                    dialogues.append({
                        'speaker': speaker,
                        'dialogue': sent.text,
                        'context': 'conversation'
                    })
        
        print("Extracted entity dialogues:")
        for dialogue in dialogues:
            print(f"  SPEAKER: {dialogue['speaker']}, DIALOGUE: {dialogue['dialogue']}")
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
        print("Stored entities and relationships in database", flush=True)

    def store_entity_dialogues(self, dialogues):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS entity_dialogues
                     (speaker TEXT, dialogue TEXT, context TEXT)''')
        for dialogue in dialogues:
            c.execute('INSERT INTO entity_dialogues VALUES (?,?,?)',
                      (dialogue['speaker'], dialogue['dialogue'], dialogue['context']))
        self.conn.commit()
        print("Stored entity dialogues in database", flush=True)

def main():
    preprocessor = DataPreprocessor()
    
    # Example book text (replace with actual book text)
    book_text = """
    "Hello, John," said Mary. "How are you today?"
    John replied, "I'm fine, thank you!"
    They both went to the park where they met with Alice.
    """
    
    preprocessor.preprocess_book(book_text)
    print("main() exit", flush=True)

if __name__ == "__main__":
    main()