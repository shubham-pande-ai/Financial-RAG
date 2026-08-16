import sys
from pathlib import Path
from synthesis.pipeline import SynthesisPipeline

def main():
    print("Testing Pipeline...")
    pipe = SynthesisPipeline()
    result = pipe.run(query="What was the net profit for Reliance in 2026?", chunks=[], symbol="RELIANCE", resolved_years=[2026])
    print("---------------------------------")
    print(f"Pipeline Mode: {result.pipeline_mode}")
    print(f"SQL Rows Used: {result.sql_rows}")
    print(f"Insights:      {result.insights}")
    print("System Prompt snippet:")
    print(result.system_prompt[:500])

if __name__ == "__main__":
    main()
