import logging
import openai

logger = logging.getLogger(__name__)
system_prompt = """
You are a helpful assistant that can convert images to Markdown format. You are given an image, and you need to convert it to Markdown format. Please output the Markdown content only, without any other text.
"""
user_prompt = """
Below is the image of one page of a document, please read the content in the image and transcribe it into plain Markdown format. Please note:
1. Identify heading levels, text styles, formulas, and the format of table rows and columns
2. Mathematical formulas should be transcribed using LaTeX syntax, ensuring consistency with the original
3. Please output the Markdown content only, without any other text.

Output Example:
```markdown
{example}
```
"""
class LLMClient:
    """
    OpenAI API compatible client class
    """
    def __init__(self, base_url:str, api_key: str, model: str):
        """
        Initialize OpenAI API client
        :param base_url: Base URL for OpenAI API
        :param api_key: OpenAI API key
        :param model: Name of the model to use
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(
                base_url=base_url,
                api_key=api_key
            )
        
    def completion(
        self,
        base64_image,
        temperature: float = 0.5,
        max_tokens: int = 8192
    ) -> str:

        # Create the message content
        user_content = [{"type": "text", "text": user_prompt}]
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
        })

        messages = []
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        else:
            messages = [
                {"role": "user", "content": user_content}
            ]
        
        try:
            response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                    )
            text = response.choices[0].message.content

            text = text.strip()
            if text.startswith("```markdown"):
                text = text[len("```markdown"):]
            if text.endswith("```"):
                text = text[:-len("```")]
            return text.strip()
            
        except Exception as e:
            logger.error(f"API request failed: {str(e)}")
            raise e
