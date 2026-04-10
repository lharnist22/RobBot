import dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config({ override: true });
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

export async function answerWithOpenAI(systemPrompt: string, userText: string): Promise<string> {
  const r = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: userText }
    ],
    temperature: 0.2
  });

  return r.choices[0]?.message?.content?.trim() ?? "";
}
