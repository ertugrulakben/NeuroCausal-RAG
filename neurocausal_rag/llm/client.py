"""
NeuroCausal RAG - LLM Client
OpenAI and other LLM provider integrations
"""

import os
from typing import Optional
import logging

from ..config import LLMConfig
from ..interfaces import ILLMClient, EvaluationResult

logger = logging.getLogger(__name__)


class LLMClient(ILLMClient):
    """
    LLM Client for answer generation and evaluation.

    Supports:
    - OpenAI (GPT-4o, GPT-4o-mini)
    - Anthropic (Claude)
    - Ollama (local models)
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._client = None

    def _get_client(self):
        """Lazy initialize the client"""
        if self._client is not None:
            return self._client

        api_key = os.environ.get(self.config.api_key_env)

        if self.config.provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key)
                logger.info("Initialized OpenAI client")
            except ImportError:
                logger.error("openai package not installed")
                raise

        elif self.config.provider == "anthropic":
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=api_key)
                logger.info("Initialized Anthropic client")
            except ImportError:
                logger.error("anthropic package not installed")
                raise

        return self._client

    def generate(self, prompt: str, context: str) -> str:
        """Generate answer given prompt and context"""
        client = self._get_client()

        system_prompt = """Sen uzman bir bilgi analisti ve dedektifsin. Görevin, verilen belgeler arasındaki GIZLI BAĞLANTILARI keşfetmek.

⚠️ KRİTİK TALİMATLAR:

1. **Multi-hop Reasoning (Çok Adımlı Çıkarım) ZORUNLU:**
   - Belge A: "X, Y'dir" diyorsa
   - Belge B: "Y, Z'dir" diyorsa
   - SONUÇ: "X, Z'dir" çıkarımını MUTLAKA yap!

2. **Kod Adı / Takma Ad Eşleştirmesi:**
   - Proje kod adlarını (örn: "Mavi Ufuk") gerçek isimlerle eşleştir
   - Gizli referansları çöz, noktaları birleştir

3. **Nedensellik Zinciri Takibi:**
   - A → B → C → D şeklindeki zincirleri takip et
   - Dolaylı bağlantıları bile yakala

4. **"Bilgi Yok" Demeden Önce:**
   - Tüm belgeleri tekrar tara
   - Dolaylı bağlantıları kontrol et
   - Parçaları birleştirmeyi dene

5. **Kanıt Göster:**
   - Her çıkarımı hangi belgeden aldığını belirt
   - "[Belge X]'e göre..." şeklinde referans ver

SEN BİR DEDEKTİFSİN. Yüzeysel bakmak yasak, derinlemesine analiz yap!"""

        user_prompt = f"""📚 BELGELER:
{context}

❓ SORU: {prompt}

🔍 TALİMAT: Yukarıdaki belgeleri bir dedektif gibi analiz et. Gizli bağlantıları, kod adlarını ve dolaylı ilişkileri keşfet. Cevabını Türkçe ver ve hangi belgelerden çıkarım yaptığını belirt."""

        if self.config.provider == "openai":
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                # temperature removed - gpt-4o-mini only supports default (1)
                max_completion_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content

        elif self.config.provider == "anthropic":
            response = client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.content[0].text

        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def generate_raw(self, prompt: str, max_tokens: int = 2000) -> str:
        """
        Generate response with raw prompt (no Q&A wrapper).
        Used for causal discovery and other analysis tasks.
        """
        client = self._get_client()

        if self.config.provider == "openai":
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens
            )
            return response.choices[0].message.content

        elif self.config.provider == "anthropic":
            response = client.messages.create(
                model=self.config.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def evaluate(self, query: str, answer: str, context: str) -> EvaluationResult:
        """Evaluate answer quality using LLM"""
        client = self._get_client()

        eval_prompt = f"""Bir RAG sisteminin cevabını değerlendir.

SORU: {query}

BAĞLAM:
{context}

CEVAP:
{answer}

Değerlendirme kriterleri:
1. Doğruluk (0-10): Cevap bağlamla tutarlı mı?
2. Bağlam Kalitesi (0-10): Bağlam soruyu cevaplamak için yeterli mi?
3. Nedensel Tutarlılık (0-10): Nedensel ilişkiler doğru mu?

Şu formatta yanıt ver:
DOGRULUK: [0-10]
BAGLAM_KALITESI: [0-10]
NEDENSEL: [0-10]
ACIKLAMA: [Kısa açıklama]"""

        if self.config.provider == "openai":
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": eval_prompt}],
                # temperature removed - gpt-4o-mini only supports default (1)
                max_completion_tokens=500
            )
            eval_text = response.choices[0].message.content
            tokens = response.usage.total_tokens

        elif self.config.provider == "anthropic":
            response = client.messages.create(
                model=self.config.model,
                max_tokens=500,
                messages=[{"role": "user", "content": eval_prompt}]
            )
            eval_text = response.content[0].text
            tokens = response.usage.input_tokens + response.usage.output_tokens

        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

        # Parse evaluation
        return self._parse_evaluation(eval_text, tokens)

    def _parse_evaluation(self, text: str, tokens: int) -> EvaluationResult:
        """Parse evaluation response"""
        lines = text.strip().split('\n')
        accuracy = 5.0
        context_quality = 5.0
        causal = 5.0
        explanation = ""

        for line in lines:
            line = line.strip()
            if line.startswith('DOGRULUK:'):
                try:
                    accuracy = float(line.split(':')[1].strip().split()[0])
                except:
                    pass
            elif line.startswith('BAGLAM_KALITESI:'):
                try:
                    context_quality = float(line.split(':')[1].strip().split()[0])
                except:
                    pass
            elif line.startswith('NEDENSEL:'):
                try:
                    causal = float(line.split(':')[1].strip().split()[0])
                except:
                    pass
            elif line.startswith('ACIKLAMA:'):
                explanation = line.split(':', 1)[1].strip()

        # Combined score
        score = (accuracy + context_quality + causal) / 3.0 / 10.0  # Normalize to 0-1

        return EvaluationResult(
            answer="",
            score=score,
            context_quality=context_quality / 10.0,
            reasoning=explanation,
            tokens_used=tokens
        )

    def get_token_count(self, text: str) -> int:
        """Estimate token count"""
        # Simple estimation: ~4 chars per token
        return len(text) // 4