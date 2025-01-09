import pandas as pd
import os
from interfaces import ISkillKnowledgeExtractor
from modules import ESCOAnalyzer, detect_language
import datetime

from config import Config

# Load configuration
configs = Config()

reports_folder_path = configs.reports_folder_path

class ESCOManager:
    def __init__(self, input_file: str, output_subfolder):
        self.input_file = input_file
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_subfolder = os.path.join(reports_folder_path, output_subfolder)
        self.output_file = os.path.join(self.output_subfolder, f"extracted_skills_{timestamp}.csv")
        self.analyzer :ISkillKnowledgeExtractor = ESCOAnalyzer()

    def run_analysis(self):
        """
        Executes the ESCO analysis and saves results.
        """
        os.makedirs(self.output_subfolder, exist_ok=True)
        df = pd.read_csv(self.input_file, encoding='utf-8')
        results = []

        for i, row in df.iterrows():
            job_description = row["Description"]
            lang = detect_language(job_description)
            extracted_data = self.analyzer.extract_skills_and_knowledge(job_description, lang)
            results.append(extracted_data)

            if (i + 1) % (len(df) // 10) == 0 or i == len(df) - 1:
                progress_df = pd.DataFrame(results)
                self.analyzer.save_progress(progress_df, self.output_subfolder)

        # Ensure the output folder is inside the 'reports' directory
        try:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            self.output_folder_path = os.path.join(reports_folder_path, f'{self.output_subfolder}_{timestamp}')
            os.makedirs(self.output_folder_path, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create output directory: {e}")
    
        # Final save
        final_df = pd.DataFrame(results)
        final_df.to_csv(self.output_file, index=False, encoding='utf-8-sig')
        self.analyzer.generate_report(self.input_file, self.output_file, self.output_subfolder)
        return f"Analysis completed. Results saved to {self.output_file}"