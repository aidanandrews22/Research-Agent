import os
from openai import OpenAI
from dotenv import load_dotenv

def test_openai_access():
    # Load environment variables
    load_dotenv()
    
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    try:
        # Make a minimal API call - just requesting 1 token
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1
        )
        print("✅ OpenAI API access successful!")
        return True
    except Exception as e:
        print(f"❌ OpenAI API access failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_openai_access() 