import os
import multiprocessing as mp
from scripts.data_processing import read_txt_files
from scripts.section_extraction import extract_sections
from scripts.embedding import EmbeddingModel
from scripts.retrieval import Retriever
from scripts.qa import QAModel
from scripts.evaluation import evaluate_model

def process_file(file_info):
    subdir = file_info["subdir"]
    filename = file_info["filename"]
    content = file_info["content"]
    
    # Extract sections based on clues
    clues = ["Death benefits", "Death benefit", "Lump sum"]
    sections = extract_sections(content, clues)
    
    return [(subdir, filename, sec) for sec in sections]

def main():
    root_dir = 'data/raw'
    gold_standard_file = 'data/gold_standard.csv'
    
    # Read and process documents
    documents = read_txt_files(root_dir)
    
    # Use multiprocessing to process files in parallel
    with mp.Pool(mp.cpu_count()) as pool:
        sections = pool.map(process_file, documents)
    
    sections = [item for sublist in sections for item in sublist]  # Flatten the list
    
    # Generate embeddings and create FAISS index
    embedding_model = EmbeddingModel()
    embeddings = embedding_model.generate_embeddings([sec[2] for sec in sections])
    index = embedding_model.initialize_faiss_index(embeddings.shape[1])
    embedding_model.add_to_index(index, embeddings)
    
    # Initialize Haystack components
    retriever = Retriever()
    retriever.write_documents([{"text": sec[2], "meta": {"subdir": sec[0], "filename": sec[1]}} for sec in sections])
    retriever.update_embeddings()
    
    qa_model = QAModel()
    qa_model.set_pipeline(retriever.retriever)
    
    # Define the function to extract information
    def extract_information(sample_paragraph):
        # Find relevant sections
        sample_embedding = embedding_model.generate_embeddings([sample_paragraph])
        D, I = index.search(sample_embedding, k=5)
        if D[0][0] > 0.3:
            return "none", "none"
        relevant_sections = [sections[idx] for idx in I[0]]
        
        # Get answers using the QA model
        answers = []
        for sec in relevant_sections:
            death_benefit, db_score = qa_model.get_answer("What is the death benefit amount?", sec[2])
            interest_rate, ir_score = qa_model.get_answer("What is the interest rate on the lump sum?", sec[2])
            answers.append((death_benefit, interest_rate, db_score, ir_score, sec[0], sec[1]))
        
        # Rank answers by score and return the best ones
        answers = sorted(answers, key=lambda x: (x[3], x[2]), reverse=True)  # Sort by interest rate score and then by death benefit score
        if answers:
            best_answer = answers[0]
            return best_answer[0], best_answer[1], best_answer[4], best_answer[5]
        return "none", "none", "none", "none"
    
    # Evaluate the model
    evaluation_results = evaluate_model(gold_standard_file, extract_information)
    print(evaluation_results)

if __name__ == "__main__":
    main()