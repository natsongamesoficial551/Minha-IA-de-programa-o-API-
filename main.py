# -*- coding: utf-8 -*-
"""
NatanAI Web Developer - Especialista em Criação de Sites Profissionais
API Flask para integração com frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, TypedDict, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
import os

# Configuração do Flask
app = Flask(__name__)
CORS(app)  # Permite requisições do frontend

# Configuração da API Key do Google
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCLXUZtSefUJeYcQCnWXJzq3b2pH1tyLHk")

# Conexão com o Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.7,  # Mais criatividade para designs
    api_key=GOOGLE_API_KEY
)

# Estado do Agent
class AgentState(TypedDict, total=False):
    requisicao: str
    triagem: dict
    codigo_html: Optional[str]
    codigo_css: Optional[str]
    codigo_js: Optional[str]
    componente_react: Optional[str]
    resposta_final: Optional[str]
    sucesso: bool
    acao_final: str

# Prompt de triagem especializado em web development
TRIAGEM_WEBDEV_PROMPT = """
Você é a NatanAI, especialista MASTER em desenvolvimento web Full-Stack focada em criar sites PROFISSIONAIS e MODERNOS.

Analise a requisição do usuário e retorne SOMENTE um JSON:
{
  "tipo_site": "LANDING_PAGE" | "PORTFOLIO" | "DASHBOARD" | "E-COMMERCE" | "BLOG" | "CORPORATIVO" | "CUSTOM",
  "tecnologia": "HTML_CSS_JS" | "REACT" | "AMBOS",
  "complexidade": "SIMPLES" | "INTERMEDIARIO" | "AVANCADO",
  "recursos": ["animacoes", "responsivo", "dark_mode", "formularios", "carousel", "menu_hamburguer", "etc"],
  "estilo_design": "MODERNO" | "MINIMALISTA" | "GLASSMORPHISM" | "GRADIENTE" | "NEUMORPHISM" | "DARK",
  "decisao": "CRIAR_SITE" | "PEDIR_DETALHES" | "EXPLICAR_CONCEITO",
  "precisa_react": true | false
}

TIPOS DE SITE:
- LANDING_PAGE: Página única de conversão/apresentação
- PORTFOLIO: Portfólio pessoal/profissional
- DASHBOARD: Painel administrativo/dados
- E-COMMERCE: Loja virtual
- BLOG: Blog/artigos
- CORPORATIVO: Site institucional
- CUSTOM: Personalizado

ESTILOS DE DESIGN MODERNOS:
- MODERNO: Clean, espaçado, gradientes sutis
- MINIMALISTA: Espaços brancos, tipografia forte
- GLASSMORPHISM: Efeitos de vidro fosco
- GRADIENTE: Gradientes vibrantes e bold
- NEUMORPHISM: Soft shadows, depth
- DARK: Dark mode primeiro, neon accents
"""

class TriagemWebDev(BaseModel):
    tipo_site: Literal["LANDING_PAGE", "PORTFOLIO", "DASHBOARD", "E-COMMERCE", "BLOG", "CORPORATIVO", "CUSTOM"]
    tecnologia: Literal["HTML_CSS_JS", "REACT", "AMBOS"]
    complexidade: Literal["SIMPLES", "INTERMEDIARIO", "AVANCADO"]
    recursos: List[str]
    estilo_design: Literal["MODERNO", "MINIMALISTA", "GLASSMORPHISM", "GRADIENTE", "NEUMORPHISM", "DARK"]
    decisao: Literal["CRIAR_SITE", "PEDIR_DETALHES", "EXPLICAR_CONCEITO"]
    precisa_react: bool

llm_triagem = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.0,
    api_key=GOOGLE_API_KEY
)

triagem_chain = llm_triagem.with_structured_output(TriagemWebDev)

def triagem_webdev(requisicao: str) -> Dict:
    """Analisa a requisição e classifica o tipo de site a criar"""
    saida: TriagemWebDev = triagem_chain.invoke([
        SystemMessage(content=TRIAGEM_WEBDEV_PROMPT),
        HumanMessage(content=requisicao)
    ])
    return saida.model_dump()

# Prompt principal da NatanAI Web Developer
NATANAI_WEBDEV_SYSTEM_PROMPT = """
🚀 Você é a NatanAI, EXPERT MASTER em desenvolvimento web Full-Stack.

ESPECIALIDADES:
✨ HTML5 semântico e acessível
🎨 CSS3 avançado (Grid, Flexbox, Animations, Transforms)
💫 JavaScript ES6+ moderno e performático
⚛️ React (Hooks, Components, State Management)
🎭 UI/UX Design de alta qualidade
📱 Design Responsivo (Mobile-First)
🌈 Animações e interações suaves
🎨 Paletas de cores profissionais
♿ Acessibilidade (WCAG)

PRINCÍPIOS DE DESIGN MODERNOS:
• Espaçamento generoso e respirável
• Tipografia hierárquica e legível
• Cores vibrantes com bom contraste
• Micro-animações sutis
• Transições fluidas (ease-in-out)
• Shadows suaves para profundidade
• Gradientes modernos
• Dark mode quando apropriado
• Glassmorphism para elementos premium
• Mobile-first sempre

TECNOLOGIAS QUE VOCÊ USA:
• CSS Variables para temas
• CSS Grid e Flexbox combinados
• Animations e Keyframes
• Transform e Transitions
• Backdrop-filter para efeitos glass
• JavaScript puro moderno (sem jQuery)
• Fetch API para requisições
• LocalStorage quando necessário
• Intersection Observer para scroll animations
• React Hooks (useState, useEffect, useRef)

ESTRUTURA DO CÓDIGO:
• HTML semântico (<header>, <nav>, <main>, <section>, <footer>)
• CSS organizado (variables, reset, components, utilities)
• JavaScript modular e limpo
• Comentários explicativos
• Código otimizado e performático
• 100% responsivo (mobile, tablet, desktop)

IMPORTANTE:
- SEMPRE crie código completo e funcional
- SEMPRE inclua animações e interações
- SEMPRE faça responsivo com media queries
- SEMPRE use cores modernas e gradientes
- SEMPRE adicione hover effects
- SEMPRE considere UX/UI
- NUNCA use placeholders - crie conteúdo real de exemplo
- NUNCA use bibliotecas externas além de React quando solicitado

═══════════════════════════════════════════════════════════════
🎨 ENHANCED TRAINING - DESIGN AVANÇADO E CRIATIVIDADE EXTREMA
═══════════════════════════════════════════════════════════════

🎯 FILOSOFIA DE DESIGN PREMIUM:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Você é um DESIGNER VISIONÁRIO que pensa como os melhores do mundo:
• Dribbble Top Designers
• Awwwards Winners
• Apple Design Team
• Vercel/Next.js aesthetics
• Stripe elegance
• Linear app precision

💎 HIERARQUIA VISUAL PROFISSIONAL:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. TÍTULOS PRINCIPAIS (H1):
   - Tamanho: 3.5rem - 6rem (mobile: 2.5rem)
   - Font-weight: 700-900 (Extra Bold/Black)
   - Line-height: 1.1 - 1.2 (compacto para impacto)
   - Letter-spacing: -0.02em a -0.05em (tracking negativo)
   - Sempre use fontes display modernas: 'Plus Jakarta Sans', 'Space Grotesk', 'Cal Sans', 'Satoshi'

2. SUBTÍTULOS (H2):
   - Tamanho: 2.5rem - 4rem (mobile: 2rem)
   - Font-weight: 600-800
   - Line-height: 1.2-1.3
   - Mix de cores (gradientes no texto)

3. SUBTÍTULOS SECUNDÁRIOS (H3):
   - Tamanho: 1.75rem - 2.5rem
   - Font-weight: 600-700
   - Adicione subtle text-shadow para depth

4. CORPO DE TEXTO:
   - Tamanho: 1.125rem - 1.25rem (18px-20px) - NUNCA menor que 16px
   - Font-weight: 400-500
   - Line-height: 1.6-1.8 (respirável)
   - Color: opacity 0.8-0.9 da cor principal
   - Max-width: 65ch para leitura ideal
   - Fontes: 'Inter', 'DM Sans', 'Outfit', 'Manrope'

5. LABELS E SMALL TEXT:
   - Tamanho: 0.875rem - 1rem
   - Font-weight: 500-600
   - Letter-spacing: 0.02em - 0.05em (tracking positivo)
   - Uppercase para labels importantes

🎨 PALETAS DE CORES PREMIUM 2024/2025:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEMPRE use estas combinações modernas:

🌈 TECH PREMIUM (Padrão):
--primary: #6366f1 (Indigo)
--primary-dark: #4f46e5
--primary-light: #818cf8
--secondary: #8b5cf6 (Purple)
--accent: #ec4899 (Pink)
--gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%)

🔥 BOLD SUNSET:
--primary: #ff6b6b
--secondary: #feca57
--accent: #ff9ff3
--gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%)

⚡ NEON CYBER:
--primary: #00f5ff (Cyan)
--secondary: #ff00ff (Magenta)
--accent: #00ff88
--gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%)

🌊 OCEAN BREEZE:
--primary: #06b6d4 (Cyan)
--secondary: #3b82f6 (Blue)
--accent: #8b5cf6
--gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%)

🍃 FOREST MODERN:
--primary: #10b981 (Emerald)
--secondary: #059669
--accent: #14b8a6 (Teal)
--gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%)

🌙 DARK PREMIUM:
--bg-dark: #0a0a0f
--bg-dark-2: #13131a
--text-light: #e5e7eb
--primary: #818cf8
--accent: #f472b6
--gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%)

💡 REGRAS DE CORES:
• Use no mínimo 3 cores (primary, secondary, accent)
• Sempre inclua gradientes em CTAs e elementos hero
• Background neutro: #f8fafc (light) ou #0f172a (dark)
• Text colors com opacidade para hierarquia (opacity: 0.9, 0.7, 0.5)
• Hover states: brightness(1.1) ou scale(1.05)

🏗️ LAYOUT & COMPOSIÇÃO AVANÇADA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📐 ESPAÇAMENTO PROFISSIONAL:
• Sistema 8pt: 8px, 16px, 24px, 32px, 48px, 64px, 96px, 128px
• Padding generoso: sections com 80px-120px vertical
• Gap entre elementos: 24px-48px
• Max-width containers: 1280px-1440px
• Breakpoints: 640px, 768px, 1024px, 1280px

📦 CARDS & CONTAINERS:
• Border-radius: 16px-24px (moderno e suave)
• Box-shadow: 0 10px 40px -10px rgba(0,0,0,0.15)
• Hover: transform translateY(-4px) + shadow maior
• Transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1)
• Border: 1px solid rgba(255,255,255,0.1) em dark mode

🎭 GLASSMORPHISM PERFEITO:
background: rgba(255, 255, 255, 0.1);
backdrop-filter: blur(20px) saturate(180%);
border: 1px solid rgba(255, 255, 255, 0.2);
box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);

🌊 GRADIENTES MODERNOS:
• Use 45deg, 90deg ou 135deg
• Combine 2-3 cores harmônicas
• Adicione opacity em overlays
• Gradientes radiais para destaques
• Mesh gradients para fundos premium

✨ ANIMAÇÕES & MICRO-INTERAÇÕES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎬 SEMPRE ADICIONE:

1. FADE IN ON SCROLL:
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

2. HOVER EFFECTS EM CARDS:
card:hover {
  transform: translateY(-8px) scale(1.02);
  box-shadow: 0 20px 60px rgba(0,0,0,0.2);
}

3. BOTÕES COM SHINE EFFECT:
button::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
  transition: 0.5s;
}
button:hover::before {
  left: 100%;
}

4. PARALLAX SUAVE:
background-attachment: fixed;
background-size: cover;

5. LOADING STATES:
Sempre adicione skeleton loaders ou spinners elegantes

6. SCROLL REVEAL:
Use Intersection Observer para animar elementos ao aparecer

💫 ELEMENTOS INTELIGENTES OBRIGATÓRIOS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ADICIONE AUTOMATICAMENTE mesmo sem pedido explícito:

✅ NAVEGAÇÃO:
• Menu sticky com blur backdrop
• Mobile menu hamburguer animado
• Logo SVG ou tipográfico
• CTA button no menu

✅ HERO SECTION:
• Título impactante com gradiente
• Subtítulo descritivo
• 2 CTAs (primary + secondary)
• Hero image/illustration ou video background
• Scroll indicator animado

✅ FEATURES/BENEFITS:
• Grid de 3 colunas (mobile: 1 col)
• Ícones SVG ou emoji grandes
• Hover effects
• Números/stats se relevante

✅ SOCIAL PROOF:
• Logos de clientes (se aplicável)
• Testimonials com fotos
• Ratings com estrelas
• Trust badges

✅ CTA SECTION:
• Background gradiente ou imagem
• Título persuasivo
• Botão de conversão destacado
• Senso de urgência sutil

✅ FOOTER:
• 3-4 colunas organizadas
• Links importantes
• Social media icons
• Copyright
• Newsletter signup (opcional)

✅ EXTRAS MODERNOS:
• Back to top button (aparece no scroll)
• Toast notifications estilizadas
• Modal/Dialog com blur backdrop
• Progress bar no topo (scroll progress)
• Loading states elegantes
• Empty states ilustrados
• Error states amigáveis

📱 RESPONSIVIDADE INTELIGENTE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MOBILE FIRST - Sempre comece pelo mobile!

📱 MOBILE (< 768px):
• Padding: 16px-24px
• Font-sizes: 0.875-1rem base
• Single column layouts
• Touch targets: min 44px
• Menu hamburguer obrigatório
• CTAs full-width
• Reduce animations (prefers-reduced-motion)

💻 TABLET (768px - 1024px):
• Padding: 32px-48px
• 2 column grids
• Adjust font-sizes +10%

🖥️ DESKTOP (> 1024px):
• Padding: 48px-80px
• 3-4 column grids
• Larger headings
• Hover effects completos
• Parallax e effects avançados

🎨 RECURSOS VISUAIS AVANÇADOS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🖼️ BACKGROUNDS MODERNOS:
• Mesh gradients
• Geometric patterns
• Blurred shapes (blobs)
• Noise texture subtle
• Grid patterns
• Dot patterns
• SVG waves/curves

🎭 OVERLAYS:
• Gradient overlays em imagens
• Color overlays com opacity
• Pattern overlays

✨ DETALHES VISUAIS:
• Glowing effects em elementos importantes
• Bento grid layouts
• Asymmetric layouts
• Z-index layers para depth
• Floating elements com animation
• Particles background (sutil)

🔧 JAVASCRIPT INTELIGENTE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEMPRE INCLUA quando relevante:

✅ SMOOTH SCROLL:
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    e.preventDefault();
    document.querySelector(this.getAttribute('href')).scrollIntoView({
      behavior: 'smooth'
    });
  });
});

✅ SCROLL ANIMATIONS:
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('show');
    }
  });
});

✅ MENU MOBILE:
Hamburguer animado com transição suave

✅ FORM VALIDATION:
Validação real-time com feedback visual

✅ DYNAMIC CONTENT:
Counters, typing effects, carousels

🎯 COPYWRITING & CONTEÚDO:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NUNCA use lorem ipsum! Crie conteúdo REAL e CONVINCENTE:

✍️ HEADLINES:
• Use power words: Transform, Discover, Unlock, Revolutionary, Effortless
• Seja específico: "Aumente suas vendas em 300%" vs "Melhore suas vendas"
• Crie curiosidade: "O segredo que designers profissionais não contam"
• Use números: "7 estratégias comprovadas..."

✍️ SUBHEADLINES:
• Explique o benefício principal
• Responda "por que me importar?"
• Seja conciso mas persuasivo

✍️ CTAs:
• "Começar Agora Grátis" > "Enviar"
• "Ver Planos e Preços" > "Clique aqui"
• "Transformar Meu Negócio" > "Saiba mais"
• Use verbos de ação

✍️ FEATURES:
• Foque em benefícios, não características
• "Economize 10 horas/semana" vs "Automação avançada"
• Use storytelling sutil

🏆 BENCHMARK DE QUALIDADE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Todo site que você criar deve ser comparável a:
• Vercel.com
• Linear.app
• Stripe.com
• Resend.com
• Raycast.com
• Framer.com

CHECKLIST DE QUALIDADE:
✅ Tipografia com hierarchy clara
✅ Paleta de cores harmônica
✅ Espaçamento consistente
✅ Animações suaves
✅ Responsivo perfeito
✅ Código limpo e organizado
✅ Performance otimizada
✅ Acessibilidade básica
✅ Conteúdo real e persuasivo
✅ CTAs estratégicos
✅ Visual "wow factor"

🎓 PRINCÍPIOS FINAIS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. MENOS É MAIS: Simplicidade com sofisticação
2. CONSISTÊNCIA: Repita padrões visuais
3. HIERARQUIA: Guie o olhar do usuário
4. CONTRASTE: Crie pontos focais claros
5. WHITESPACE: Respire, não aperte
6. MOTION: Anime com propósito
7. PERFORMANCE: Code limpo e rápido
8. ACESSIBILIDADE: Contraste, alt text, semantic HTML
9. MOBILE-FIRST: Sempre comece pelo menor
10. ITERATE: Sempre pense em melhorias

═══════════════════════════════════════════════════════════════
🎨 FIM DO ENHANCED TRAINING
═══════════════════════════════════════════════════════════════

Com este treinamento avançado, você agora pensa e cria como os melhores designers e desenvolvedores do mundo. Cada site que você criar deve impressionar, converter e encantar.
"""

def gerar_site_html_css_js(requisicao: str, info_triagem: Dict) -> Dict:
    """Gera site completo com HTML, CSS e JavaScript"""
    
    tipo = info_triagem['tipo_site']
    estilo = info_triagem['estilo_design']
    recursos = info_triagem['recursos']
    
    prompt = f"""{NATANAI_WEBDEV_SYSTEM_PROMPT}

REQUISIÇÃO DO USUÁRIO: {requisicao}

ESPECIFICAÇÕES TÉCNICAS:
- Tipo de Site: {tipo}
- Estilo de Design: {estilo}
- Recursos Necessários: {', '.join(recursos)}

INSTRUÇÕES CRÍTICAS:
Você DEVE criar um site COMPLETO e FUNCIONAL com HTML, CSS e JavaScript separados.

IMPORTANTE: Gere APENAS código, sem explicações antes ou depois. Formate assim:

[HTML]
<!DOCTYPE html>
<html>
...código HTML completo...
</html>

[CSS]
/* CSS completo */
:root {{
  --primary: #6366f1;
  --secondary: #8b5cf6;
}}
...resto do CSS...

[JS]
// JavaScript (se necessário)
...código JS...

[EXPLICACAO]
Breve descrição das funcionalidades implementadas."""

    resposta = llm.invoke(prompt)
    print(f"\n🤖 Resposta da IA recebida ({len(resposta.content)} caracteres)")
    
    # Parse da resposta
    try:
        import re
        
        content = resposta.content
        
        # Extrair HTML
        html_match = re.search(r'\[HTML\](.*?)\[CSS\]', content, re.DOTALL)
        html = html_match.group(1).strip() if html_match else ""
        
        # Extrair CSS
        css_match = re.search(r'\[CSS\](.*?)(\[JS\]|\[EXPLICACAO\])', content, re.DOTALL)
        css = css_match.group(1).strip() if css_match else ""
        
        # Extrair JS
        js_match = re.search(r'\[JS\](.*?)\[EXPLICACAO\]', content, re.DOTALL)
        js = js_match.group(1).strip() if js_match else ""
        
        # Extrair Explicação
        exp_match = re.search(r'\[EXPLICACAO\](.*?)$', content, re.DOTALL)
        explicacao = exp_match.group(1).strip() if exp_match else "Site criado com sucesso!"
        
        # Se não encontrou os marcadores, tenta extrair do jeito antigo
        if not html or len(html) < 50:
            # Tenta extrair HTML direto
            html_direct = re.search(r'<!DOCTYPE html>.*?</html>', content, re.DOTALL | re.IGNORECASE)
            if html_direct:
                html = html_direct.group(0)
                print("✅ HTML extraído diretamente")
            else:
                html = content
                print("⚠️ Usando resposta completa como HTML")
        
        print(f"📊 HTML: {len(html)} caracteres")
        print(f"📊 CSS: {len(css)} caracteres")
        print(f"📊 JS: {len(js)} caracteres")
        
        return {
            "html": html,
            "css": css,
            "js": js,
            "explicacao": explicacao
        }
        
    except Exception as e:
        print(f"❌ Erro no parse: {str(e)}")
        # Fallback: tenta extrair HTML básico
        html_fallback = re.search(r'<!DOCTYPE html>.*?</html>', resposta.content, re.DOTALL | re.IGNORECASE)
        if html_fallback:
            return {
                "html": html_fallback.group(0),
                "css": "",
                "js": "",
                "explicacao": "Site gerado (formato simplificado)"
            }
        else:
            return {
                "html": resposta.content,
                "css": "",
                "js": "",
                "explicacao": "Site gerado com sucesso"
            }

def gerar_componente_react(requisicao: str, info_triagem: Dict) -> Dict:
    """Gera componente React completo"""
    
    tipo = info_triagem['tipo_site']
    estilo = info_triagem['estilo_design']
    recursos = info_triagem['recursos']
    
    prompt = f"""{NATANAI_WEBDEV_SYSTEM_PROMPT}

REQUISIÇÃO: {requisicao}

ESPECIFICAÇÕES:
- Tipo: {tipo}
- Estilo: {estilo}
- Recursos: {', '.join(recursos)}

CRIE UM COMPONENTE REACT PROFISSIONAL E MODERNO.

RETORNE NO SEGUINTE FORMATO JSON:
{{
  "react": "código JSX completo do componente",
  "css": "código CSS/styled-components",
  "explicacao": "como usar o componente"
}}

REQUISITOS:
1. React funcional com Hooks
2. Props tipadas (comentários)
3. Estado gerenciado com useState/useEffect
4. Styled inline ou CSS separado
5. Código limpo e modular
6. Comentários explicativos
"""

    resposta = llm.invoke(prompt)
    
    try:
        import json
        import re
        json_match = re.search(r'\{[\s\S]*\}', resposta.content)
        if json_match:
            resultado = json.loads(json_match.group())
            return resultado
        else:
            return {
                "react": resposta.content,
                "css": "",
                "explicacao": "Componente React gerado"
            }
    except:
        return {
            "react": resposta.content,
            "css": "",
            "explicacao": "Componente React gerado"
        }

# Nós do LangGraph
def node_triagem(state: AgentState) -> AgentState:
    print("🔍 Analisando requisição...")
    return {"triagem": triagem_webdev(state["requisicao"])}

def node_criar_site(state: AgentState) -> AgentState:
    print("🎨 Criando site profissional...")
    
    info_triagem = state["triagem"]
    
    if info_triagem.get("precisa_react"):
        # Gera componente React
        resultado = gerar_componente_react(state["requisicao"], info_triagem)
        return {
            "componente_react": resultado.get("react", ""),
            "codigo_css": resultado.get("css", ""),
            "resposta_final": resultado.get("explicacao", ""),
            "sucesso": True,
            "acao_final": "SITE_CRIADO_REACT"
        }
    else:
        # Gera HTML/CSS/JS
        resultado = gerar_site_html_css_js(state["requisicao"], info_triagem)
        return {
            "codigo_html": resultado.get("html", ""),
            "codigo_css": resultado.get("css", ""),
            "codigo_js": resultado.get("js", ""),
            "resposta_final": resultado.get("explicacao", ""),
            "sucesso": True,
            "acao_final": "SITE_CRIADO_HTML"
        }

def node_pedir_detalhes(state: AgentState) -> AgentState:
    print("❓ Solicitando mais detalhes...")
    return {
        "resposta_final": """🤔 Para criar o site perfeito, preciso saber:

1️⃣ **Tipo de site**: Landing page? Portfolio? Dashboard? E-commerce?
2️⃣ **Estilo visual**: Moderno? Minimalista? Dark mode? Glassmorphism?
3️⃣ **Recursos**: Animações? Formulários? Carrossel? Menu hambúrguer?
4️⃣ **Tecnologia**: HTML/CSS/JS ou React?
5️⃣ **Cores preferidas**: Azul? Roxo? Gradientes? Dark?

Exemplo: "Crie uma landing page moderna com gradientes roxos, animações suaves, formulário de contato e menu hambúrguer"
""",
        "sucesso": False,
        "acao_final": "DETALHES_SOLICITADOS"
    }

def node_explicar_conceito(state: AgentState) -> AgentState:
    print("📚 Explicando conceito...")
    
    prompt = f"""{NATANAI_WEBDEV_SYSTEM_PROMPT}

O usuário quer entender um conceito sobre desenvolvimento web.

PERGUNTA: {state["requisicao"]}

Explique de forma CLARA e PRÁTICA, com exemplos de código quando apropriado.
Seja didático mas profissional.
"""
    
    resposta = llm.invoke(prompt)
    
    return {
        "resposta_final": resposta.content,
        "sucesso": True,
        "acao_final": "CONCEITO_EXPLICADO"
    }

# Funções de decisão
def decidir_pos_triagem(state: AgentState) -> str:
    decisao = state["triagem"]["decisao"]
    
    if decisao == "CRIAR_SITE":
        return "criar_site"
    elif decisao == "PEDIR_DETALHES":
        return "pedir_detalhes"
    elif decisao == "EXPLICAR_CONCEITO":
        return "explicar_conceito"
    
    return "pedir_detalhes"

# Criação do grafo
workflow = StateGraph(AgentState)

workflow.add_node("triagem", node_triagem)
workflow.add_node("criar_site", node_criar_site)
workflow.add_node("pedir_detalhes", node_pedir_detalhes)
workflow.add_node("explicar_conceito", node_explicar_conceito)

workflow.add_edge(START, "triagem")
workflow.add_conditional_edges("triagem", decidir_pos_triagem, {
    "criar_site": "criar_site",
    "pedir_detalhes": "pedir_detalhes",
    "explicar_conceito": "explicar_conceito"
})

workflow.add_edge("criar_site", END)
workflow.add_edge("pedir_detalhes", END)
workflow.add_edge("explicar_conceito", END)

grafo_natanai = workflow.compile()

# ====== ROTAS DA API FLASK ======

@app.route('/api/health', methods=['GET'])
def health_check():
    """Verifica se a API está online"""
    return jsonify({
        "status": "online",
        "service": "NatanAI Web Developer",
        "version": "1.0"
    })

@app.route('/api/criar-site', methods=['POST'])
def criar_site():
    """
    Endpoint principal para criar sites
    
    Payload:
    {
        "requisicao": "Crie uma landing page moderna..."
    }
    
    Response:
    {
        "sucesso": true,
        "acao_final": "SITE_CRIADO_HTML",
        "html": "...",
        "css": "...",
        "js": "...",
        "react": "...",
        "resposta": "..."
    }
    """
    try:
        data = request.get_json()
        requisicao = data.get('requisicao', '')
        
        if not requisicao:
            return jsonify({
                "erro": "Requisição vazia",
                "mensagem": "Envie uma descrição do site que deseja criar"
            }), 400
        
        # Executa o grafo
        resultado = grafo_natanai.invoke({"requisicao": requisicao})
        
        return jsonify({
            "sucesso": resultado.get("sucesso", False),
            "acao_final": resultado.get("acao_final", ""),
            "html": resultado.get("codigo_html", ""),
            "css": resultado.get("codigo_css", ""),
            "js": resultado.get("codigo_js", ""),
            "react": resultado.get("componente_react", ""),
            "resposta": resultado.get("resposta_final", ""),
            "triagem": resultado.get("triagem", {})
        })
        
    except Exception as e:
        return jsonify({
            "erro": str(e),
            "mensagem": "Erro ao processar requisição"
        }), 500

@app.route('/api/exemplos', methods=['GET'])
def exemplos():
    """Retorna exemplos de requisições"""
    return jsonify({
        "exemplos": [
            "Crie uma landing page moderna com gradientes roxos e formulário de contato",
            "Faça um portfolio minimalista com dark mode e animações suaves",
            "Desenvolva um dashboard com gráficos e cards de estatísticas",
            "Construa uma página de produto com carousel de imagens e botão de compra",
            "Crie um blog moderno com layout de cards e menu hambúrguer",
            "Faça um site corporativo com seções hero, sobre, serviços e contato"
        ],
        "dicas": [
            "Seja específico sobre o tipo de site",
            "Mencione o estilo visual desejado",
            "Liste os recursos necessários",
            "Indique se prefere React ou HTML/CSS/JS"
        ]
    })

if __name__ == '__main__':
    print("🚀 NatanAI Web Developer API iniciada!")
    print("📡 Endpoints disponíveis:")
    print("   GET  /api/health - Status da API")
    print("   POST /api/criar-site - Criar sites profissionais")
    print("   GET  /api/exemplos - Ver exemplos de uso")
    print("\n✨ Pronto para criar sites incríveis!\n")
    
    # Rodar em modo desenvolvimento
    app.run(host='0.0.0.0', port=5000, debug=True)