# pdf-rag

A hello-world of RAG application using LangChain

Loads a PDF file into an in-memory vector store and allows you to ask questions about it.

```
bun run index.ts tracemonkey.pdf

Loading PDF file path: /Users/dev/dev/pdf-rag/tracemonkey.pdf
Loaded tracemonkey.pdf with 14 documents

Creating embeddings...
Created embeddings stored in an in-memory vector store


📚 Ask questions about your PDF (type '/bye' to exit)

Can you give me a technical summary of this pdf?

Thinking... 🤔

🤖 Answer: The PDF discusses a technique for efficiently running dynamic languages by recording hot traces and generating type-specialized native code, focusing on aggressively inlined loops. It details the generation of nested traces to minimize code duplication and describes a trace compiler that optimizes code in two linear passes. Experimental results indicate that most loops are entered with limited value type combinations, allowing for effective native code execution. 

Ask another question or type '/bye' to exit

can you expand on how most loops are entered with limited value type combinations ? What does that mean exactly?

Thinking... 🤔

🤖 Answer: Most loops are entered with limited value type combinations because the initial assumption is that variables will maintain a consistent type, often starting as integers. If a variable unexpectedly changes to a different type, such as a double, it creates type instability, leading to mis-speculation. To address this, the system consults an "oracle" to ensure that the correct type is used for future iterations, allowing for type-stable execution. 

Ask another question or type '/bye' to exit

/bye

Goodbye! 👋
```