# AI Fictional Character Persona
Input your favorite fictional character, and chat with them like friends!

1_preprocessData.py
what does it do?
1. set up the space, working direction
2. set up ollama (llama)
3. tokenized (split the text)
4. extract entities & relationship from text (using LLM)
5. extract entities & dialogues from text (using LLM)
6. store entities+relationship, and store entities+dialogues

2_EntitiesRelationshipGraph.py
what does it do?
1. match the the "same"/ similar "entities" -> check synonym, and create the entities based relationship

3_EntitiesDialoguesGraph.py
1. match the the "same"/ similar "entities" -> check synonym, and create the entities based dialogues

4_SearchingGraph.py

how to start ollama:
ollama serve &