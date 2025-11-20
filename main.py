import os
import dspy
from dotenv import load_dotenv
from agent.brand_agent import BrandAgent
from agent.category_agent import CategoryAgent
from agent.reach_agent import ReachEstimatorAgent
from agent.brand_lift_agent import BrandLiftAgent

load_dotenv()

# Configure once globally
dspy.configure(lm=dspy.LM("ollama/phi3"))


# prompts = [
#     'Which is better for kitchen ware: Solimo or Cello?',
#     'Can you summarize the most discussed topics in entertainment news this week?',
#     'Are Samsung and LG making their media products more sustainable or energy-efficient?',
# 'Are there any upcoming deals or discounts on smart TVs from Samsung or Philips?',
# 'Can you compare the smart features of Samsung Smart TVs and Sony Bravia?',
# 'Can you help me choose between Samsung’s The Frame and Sony’s A80L for a living room setup?',
# 'Can you help me write a script for a podcast episode on media literacy?',
# 'Can you list the top-rated 4K projectors for home cinema from brands like Epson, LG, and BenQ?',
# 'Can you summarize the most discussed topics in entertainment news this week?',
# 'How are augmented reality and virtual reality being used in media storytelling?',
# 'How can we detect deepfakes and AI-generated misinformation in news media?',
# 'How do algorithms on platforms like YouTube and Netflix influence what people watch?',
# 'How does AI upscaling work in Samsung TVs and is it better than LG’s version?',
# 'I am getting sore eyes from watching tv. What can I do about this?',
# 'Is the LG C3 OLED worth the upgrade from the LG CX?',
# 'What are some innovative formats for interactive documentaries?',
# 'What are the current trends in streaming services and how are they affecting traditional TV?',
# 'What are the ethical concerns around using AI to generate news articles?',
# 'What are the latest trends in smart TV technology from brands like Samsung and TCL?',
# 'What are the most common issues users report with LG soundbars?',
# 'What are the pros and cons of buying a soundbar vs a full home theater system?',
# 'What impact is AI having on journalism and news production?',
# 'What voice assistants are compatible with Sony TVs and how do they enhance media control?',
# 'Which is better for home entertainment: Samsung QLED vs LG OLED TVs?',
# 'Which is the best 65-inch TV under €1000 for watching sports?',
# 'Which tablet is best for streaming and media editing: iPad Pro or Samsung Galaxy Tab?',
# 'Why do people prefer short-form video content like TikTok over long-form formats?',
# ]

prompts = ['Is Solimo better or Cello']

def main():
    brand_agent = BrandAgent()
    category_agent = CategoryAgent()
    # reach_agent = ReachEstimatorAgent()
    # lift_agent = BrandLiftAgent()

    for prompt in prompts:
        brands = brand_agent.extract_brands(prompt)
        category = category_agent.extract_category(prompt)
        print(f"\nPrompt: {prompt}")
        print(f"→ Brands: {brands}")
        print(f"→ Category: {category}")

if __name__ == "__main__":
    main()