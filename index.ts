import { config } from "dotenv";
import { existsSync } from "fs";
import { resolve, basename } from "path";
import type { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { PromptTemplate } from "@langchain/core/prompts";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";



const prompt = `You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:`

async function main() {
  // Load environment variables
  config();

  // Verify API key is present
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY is required in .env file");
  }

  // Get command line arguments
  const args = process.argv.slice(2);

  if (args.length === 0) {
    throw new Error("Please provide a PDF file path as an argument");
  }

  const pdfPath = resolve(args[0]);

  // Verify file exists and has .pdf extension
  if (!existsSync(pdfPath)) {
    throw new Error(`File not found: ${pdfPath}`);
  }

  if (!pdfPath.toLowerCase().endsWith(".pdf")) {
    throw new Error("File must be a PDF");
  }

  console.log(`Loading PDF file path: ${pdfPath}`);
  // Your PDF processing logic will go here
  const docs = await loadPDF(pdfPath);
  console.log(`Loaded ${basename(pdfPath)} with ${docs.length} documents`);

  const llm = new ChatOpenAI({
    model: "gpt-4o-mini",
    temperature: 0,
    apiKey: process.env.OPENAI_API_KEY,
  });


  console.log(`Creating embeddings...`);
  const vectorStore = await createEmbeddings(docs);
  console.log(`Created embeddings stored in an in-memory vector store`);

  const promptTemplate = PromptTemplate.fromTemplate(prompt);

  const InputStateAnnotation = Annotation.Root({
    question: Annotation<string>,
  });

  const StateAnnotation = Annotation.Root({
    question: Annotation<string>,
    context: Annotation<Document[]>,
    answer: Annotation<string>,
  });



  // Define application steps
  const retrieve = async (state: typeof InputStateAnnotation.State) => {
    const retrievedDocs = await vectorStore.similaritySearch(state.question);
    return { context: retrievedDocs };
  };

  const generate = async (state: typeof StateAnnotation.State) => {
    const docsContent = state.context.map((doc) => doc.pageContent).join("\n");
    const messages = await promptTemplate.invoke({
      question: state.question,
      context: docsContent,
    });
    const response = await llm.invoke(messages);
    return { answer: response.content };
  };

  // Compile application and test
  const graph = new StateGraph(StateAnnotation)
    .addNode("retrieve", retrieve)
    .addNode("generate", generate)
    .addEdge("__start__", "retrieve")
    .addEdge("retrieve", "generate")
    .addEdge("generate", "__end__")
    .compile();

  const readline = Bun.stdin.stream();
  const decoder = new TextDecoder();
  
  console.log("Ask questions about your PDF (type '/bye' to exit)");
  
  for await (const chunk of readline) {
    const question = decoder.decode(chunk).trim();
    
    if (question.toLowerCase() === '/bye') {
      console.log("Goodbye!");
      break;
    }
    
    if (question.length === 0) continue;
    
    const inputs = { question };
    try {
      const result = await graph.invoke(inputs);
      console.log("\nAnswer:", result.answer);
    } catch (error) {
      console.error("Error processing question:", error);
    }
    console.log("\nAsk another question or type '/bye' to exit");
  }
}

async function loadPDF(path: string) {
  const loader = new PDFLoader(path);

  const docs = await loader.load();
  return docs;
}

async function createEmbeddings(docs: Document[]) {
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const allSplits = await textSplitter.splitDocuments(docs);

  const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-large",
    apiKey: process.env.OPENAI_API_KEY,
    openAIApiKey: process.env.OPENAI_API_KEY,
    verbose: true,
  });

  const vectorStore = new MemoryVectorStore(embeddings);

  await vectorStore.addDocuments(allSplits);

  return vectorStore;
}

main();
