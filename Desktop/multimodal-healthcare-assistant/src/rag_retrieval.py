class RAGRetrieval:
    def get_context(self, text):
        print("Fetching context (dummy)...")
        return {
            "patient_history": "None",
            "relevant_info": "General medical info"
        }