# -*- coding: utf-8 -*-
# AIER-IEEE 29148/2018 - Assistente de Elicitação e Especificação de Requisitos
# Fluxo: ChatGPT-like + RAG em PDF da ISO 29148 + Conflito entre stakeholders + Export PDF

import streamlit as st
from dotenv import load_dotenv
import os
import io
import tempfile
from datetime import datetime
from fpdf import FPDF

# LangChain / OpenAI / RAG
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# PDF
from pypdf import PdfReader

# -----------------------------
# Configuração inicial
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("Defina a variável de ambiente OPENAI_API_KEY no seu .env")
    st.stop()

# Modelo de linguagem
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# -----------------------------
# Utilitários
# -----------------------------
@st.cache_data(show_spinner=False)
def _read_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text = []
    for page in reader.pages:
        try:
            text.append(page.extract_text() or "")
        except Exception:
            text.append("")
    return "\n".join(text)

@st.cache_resource(show_spinner=True)
def _build_vectorstore_from_iso(iso_pdf_bytes: bytes):
    """Constroi FAISS com embeddings a partir do PDF da ISO (RAG)."""
    raw_text = _read_pdf_text(iso_pdf_bytes)
    if not raw_text.strip():
        raise RuntimeError("Não foi possível extrair texto do PDF da ISO. Verifique o arquivo.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    docs = splitter.create_documents([raw_text])
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vs = FAISS.from_documents(docs, embeddings)
    return vs

def _retrieve_norma_context(vs: FAISS, query: str, k: int = 4) -> str:
    if vs is None:
        return ""
    docs = vs.similarity_search(query, k=k)
    trechos = []
    for i, d in enumerate(docs, 1):
        trechos.append(f"[Trecho {i}]\n{d.page_content.strip()}")
    return "\n\n".join(trechos)

def _now_date():
    return datetime.now().strftime("%Y-%m-%d")

def _hash(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# -----------------------------
# Prompts (embutidos)
# -----------------------------
PROMPT_ELICITAR = ChatPromptTemplate.from_messages([
    ("system",
     """A partir de agora, você atuará como um Engenheiro Especialista em Especificação de Requisitos de software, com profundo conhecimento 
     prático e teórico em especificar requisitos seguindo rigorosamente a norma ISO/IEC/IEEE 29148:2018. Seu papel e:
     
- Ler atentamente uma transcrição de entrevista OU uma descrição de sistema do stakeholder.
- Especificar requisitos funcionais (RF) e requisitos não funcionais (RNF) necessário, 
apropriado, sem ambiguidade, completo, singular, viável, verificável, correto, conforme separadamente;
- “A partir de agora, sempre que especificar requisitos use a sintaxe: [Sujeito] deve [verbo de ação] [objeto] [restrição]. Como
mostra o exemplo: O sistema deve gerar relatório  mensal em formato PDF:
Sujeito = O Sistema
Termo obrigatorio = deve
Verbo de acao = gerar
Objeto = relatorio mensal
Restricao = em formato PDF.
- Use a voz ativa: evite usar a voz passiva, como "é necessário que".
- Preste muita atenção aos detalhes de verbo de ação. Quando identificar em um requisito mais de um verbo de ação ou entidade, 
separa as ações em requisitos correspondentes.
- Requisitos funcionais e não funcionais devem iniciar com verbos de ação (ex.: registrar, gerar, exibir).
- Use “deve” para indicar obrigatoriedade. Ex: “O sistema deve registrar...”. 
-Fornece detalhes relevantes. Exemplo: O sistema deve permitir que o administrador cadastre usuários informando nome completo, CPF, e-mail válido, senha com no mínimo 8 caracteres, contendo letras maiúsculas, minúsculas e números.
- Sempre especifique claramente cada ator (quem realiza a ação) em cada requisito.
- Nunca junte múltiplas ações no mesmo requisito funcional. Cada ação = um requisito separado, com numeração distinta (RF01, RF02, ...).
- Cada requisito funcional deve conter apenas uma ação clara, mensurável e verificável.
- Sem dependência externa: [Sujeito] [Ação] [Objeto] [Restrição da Ação].
- Com dependência/condição: [Condição], [Sujeito] deve [Ação] [Objeto] [Restrição].
- RNF: cobrir desempenho, segurança, usabilidade, disponibilidade, confiabilidade, manutenibilidade e portabilidade.
- Exemplos:
  - Desempenho: “O sistema deve processar 95% dos pedidos em até 2 segundos.”
  - Portabilidade: “O sistema deve funcionar em Chrome/Firefox/Edge e ser compatível com Windows e Linux.”
- Para cada RF e RNF, inclua atributos (em formato vertical):
ID, Descrição, Justificativa, Versão, Prioridade, Risco, Dificuldade, Referência de Origem, Data de Criação, Proprietário.
- Categorize os requisitos por tema (ex.: inscrição, frequência, notificações, gerenciamento).

Você tem acesso a trechos relevantes da ISO (RAG) no campo {contexto_norma}.
Baseie-se fortemente neles para a aderência normativa.
Responda SOMENTE com os requisitos (RF e RNF) nos blocos especificados, sem comentários adicionais.
"""),
    ("user",
     """Contexto normativo da ISO (RAG):
{contexto_norma}

Entrada do Stakeholder:
{texto}

Gere requisitos RF e RNF completos com os atributos exigidos.
Data de hoje: {hoje}
Proprietário padrão (o nome do stakeholder que responde as perguntas na entrevista).
Proprietário padrão (se não especificado): Engenheiro de Requisitos
""")
])

PROMPT_VALIDAR = ChatPromptTemplate.from_messages([
    ("system",
     """ Dada uma lista de requisitos (RF/RNF) especificados:
1-A partir de agora, preste muita atenção aos detalhes do atributo "Descrição" para detectar Ambiguidade de requisito especificado procure:
-Termos vagos/ambíguos como ("fácil", "eficiente", "melhor", "adequado", "rápido", "intuitivo") que
 precisam de esclarecimento conforme critérios de qualidade de requisitos seguindo a norma ISO/IEC/IEEE 29148:2018, apenas no atributo "Descrição" do requisito;
-Qualquer trecho que possa ser interpretado de mais de uma forma;
- Composição atômica incorreta contendo ("e", "ou") sem critério.
- Para cada ambiguidade encontrada, explique:
    -O trecho ambíguo.
    -Por que é ambíguo.
    -Como reformular para torná-lo claro e inequívoco.
2- Analise o atributo "Descrição" somente para detectar incompletude, nele deve verificar:
- O requisito é completo;
- Define suficientemente todas as condições e partes envolvidas.
- Carece de uma informaçao para completar o contexto.
Se estiver incompleto, liste:
    -O que está faltando baseando-se na sintaxe: [Sujeito] deve [verbo de ação] [objeto] [restrição]
    -Perguntas específicas que precisam ser respondidas para torná-lo completo.
- Gere UMA pergunta objetiva uma por vez que precisam ser respondidas para torná-lo completo.
"""),
    ("user",
     """Requisitos atuais:
{requisitos}

Gere UMA pergunta objetiva, clara e verificável, sobre um único requisito.
""")
])

# >>>> REFINAR ALTERADO: retorna ESPECIFICAÇÃO COMPLETA CONSOLIDADA <<<<
PROMPT_REFINAR = ChatPromptTemplate.from_messages([
    ("system",
     """Você deve fazer perguntas até que o requisito nao tenha palavra ambigua no atributo e seja completo:
-Reescreva/adicione/atualize os requisitos (RF/RNF) incorporando a RESPOSTA do stakeholder à pergunta objetiva, mantendo:
- Estilo e formato (IDs RF/RNF e seus atributos).
- A numeração consistente (gerar novos IDs se surgirem novos requisitos).
- Conformidade com a ISO/IEC/IEEE 29148:2018.
IMPORTANTE:
- Retorne a ESPECIFICAÇÃO COMPLETA CONSOLIDADA (todos os RF e RNF, já atualizados).
- Substitua versões antigas pelos refinamentos (não duplique IDs).
- Se atualizar um requisito, incremente o campo Versão e mantenha o mesmo ID.
- Ao final de cada requisito atualizado, acrescente: "Atualizado a partir da pergunta: {pergunta} em {hoje}".
- Responda apenas com os requisitos consolidados (completos), sem comentários adicionais.
"""),
    ("user",
     """Requisitos atuais:
{requisitos}

Pergunta que foi feita:
{pergunta}

Resposta do stakeholder:
{resposta}

Atualize os requisitos conforme necessário e RETORNE A ESPECIFICAÇÃO COMPLETA.
Data de hoje: {hoje}
""")
])

PROMPT_CONFLITOS = ChatPromptTemplate.from_messages([
    ("system",
     """Você deve para cada requisito no conjunto verificar se há inconsistência ou conflito com outros requisitos no atributi "Descrição":
-Analise requisitos (RF/RNF) oriundos de 2+ stakeholders;
Um conflito pode ser contradição de termos, objetivos incompatíveis ou condições mutuamente exclusivas.
Retorne:
- Requisitos que estão consistentes.
- Pares de requisitos em conflito, com explicação de por que há inconsistência.
- Sugestões de como resolver cada inconsitência.
Para cada conflito encontrado, retorne neste formato:
[Conflito N]
Tipo: <tipo>
Requisitos em conflito: <IDs ou trechos resumidos>
Explicação breve:
Sugerir encaminhamento (pergunta objetiva única):
Não invente conflito; só reporte quando houver evidência no texto.
"""),
    ("user",
     """Requisitos consolidados (podem incluir marcações de origem de stakeholder):
{requisitos}

Liste todos os conflitos encontrados seguindo o formato solicitado.
Se não houver, diga "Nenhum conflito encontrado." sem comentários adicionais.
""")
])

# -----------------------------
# Cadeias
# -----------------------------
parser = StrOutputParser()

def chain_elicitar(modelo: ChatOpenAI, contexto_norma: str, texto: str) -> str:
    chain = PROMPT_ELICITAR | modelo | parser
    return chain.invoke({"contexto_norma": contexto_norma, "texto": texto, "hoje": _now_date()})

def chain_validar(modelo: ChatOpenAI, requisitos_texto: str) -> str:
    chain = PROMPT_VALIDAR | modelo | parser
    return chain.invoke({"requisitos": requisitos_texto})

def chain_refinar(modelo: ChatOpenAI, requisitos_texto: str, pergunta: str, resposta: str) -> str:
    chain = PROMPT_REFINAR | modelo | parser
    return chain.invoke({
        "requisitos": requisitos_texto,
        "pergunta": pergunta,
        "resposta": resposta,
        "hoje": _now_date()
    })

def chain_conflitos(modelo: ChatOpenAI, requisitos_texto: str) -> str:
    chain = PROMPT_CONFLITOS | modelo | parser
    return chain.invoke({"requisitos": requisitos_texto})

# -----------------------------
# Export PDF
# -----------------------------
def exportar_requisitos_pdf(requisitos_texto: str) -> str:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 8, "Requisitos Especificados (AIER-IEEE 29148/2018)", ln=True, align="C")
    pdf.ln(4)

    pdf.set_font("Arial", size=11)
    for linha in requisitos_texto.splitlines():
        if linha.strip().startswith(("- ID:", "- Descrição:", "- Justificativa:", "- Versão:", "- Prioridade:",
                                     "- Risco:", "- Dificuldade:", "- Referência de Origem:", "- Data de Criação:",
                                     "- Proprietário:")):
            pdf.set_font("Arial", style="B", size=11)
            pdf.multi_cell(0, 6, linha)
            pdf.set_font("Arial", size=11)
        else:
            pdf.multi_cell(0, 6, linha)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        return tmp.name

# -----------------------------
# UI (Streamlit)
# -----------------------------
st.set_page_config(page_title="AIER-IEEE 29148/2018", page_icon="", layout="wide")

st.markdown(
    """
    <h2 style='text-align:center;'>AIER-IEEE 29148/2018</h2>
    <p style='text-align:center;'>Especificação de Requisitos com base na norma ISO 29148:2018</p>
    """,
    unsafe_allow_html=True
)

# Sidebar: ISO PDF
with st.sidebar:
    st.subheader("Base Normativa-RAG")
    iso_file = st.file_uploader("Carregue o PDF da ISO/IEC/IEEE 29148:2018", type=["pdf"])
    top_k = st.slider("Recuperacao da norma (k)", 2, 8, 4)
    st.markdown("---")
    st.caption("Dica: deixe este PDF carregado; toda resposta sera baseada na norma.")

# Estado
if "historico" not in st.session_state:
    st.session_state.historico = []
if "requisitos_texto" not in st.session_state:
    st.session_state.requisitos_texto = ""
if "etapa" not in st.session_state:
    st.session_state.etapa = "especificar"
if "ultima_pergunta" not in st.session_state:
    st.session_state.ultima_pergunta = ""
if "vs" not in st.session_state:
    st.session_state.vs = None
if "last_input_hash" not in st.session_state:
    st.session_state.last_input_hash = None

# Constrói vetorstore
if iso_file is not None and st.session_state.vs is None:
    try:
        st.session_state.vs = _build_vectorstore_from_iso(iso_file.read())
        st.success("Base da ISO carregada e indexada com sucesso")
    except Exception as e:
        st.error(f"Erro ao processar PDF da ISO: {e}")

# Modelo LLM
modelo = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY, temperature=0.2)

# -------- Entrada unificada (com form + hash anti-duplicação) --------
st.markdown("#### Entrada de texto")
with st.form("form_elicitacao", clear_on_submit=False):
    tipo_texto = st.radio(
        "Selecione o tipo de entrada:",
        ["Descrição do sistema", "Transcrição de entrevista"],
        key="tipo_texto"
    )
    descricao = st.text_area("Digite ou cole o texto aqui", height=200, key="descricao")
    submitted = st.form_submit_button("Especifcar requisitos")

if submitted and descricao.strip():
    referencia_origem = f"{tipo_texto} ({_now_date()})"
    input_hash = _hash(referencia_origem + "\n" + descricao)

    if input_hash != st.session_state.last_input_hash:
        st.chat_message("user").write(f"**{referencia_origem}:**\n\n{descricao}")
        st.session_state.historico.append(("user", f"{referencia_origem}: {descricao}"))

        contexto = _retrieve_norma_context(
            st.session_state.vs,
            "Especificação de requisitos segundo ISO 29148:2018",
            k=top_k
        ) if st.session_state.vs else ""

        with st.spinner("Especificando requisitos"):
            requisitos = chain_elicitar(modelo, contexto, descricao)

        # SUBSTITUI (não anexar)
        st.session_state.requisitos_texto = requisitos.strip()
        st.session_state.etapa = "validar"
        st.session_state.last_input_hash = input_hash

        st.chat_message("assistant").write(st.session_state.requisitos_texto)
        st.session_state.historico.append(("assistant", st.session_state.requisitos_texto))
    else:
        st.info("Mesmo conteúdo já processado — nada foi refeito.")

# -------- Validação --------
if st.session_state.etapa == "validar" and st.session_state.requisitos_texto:
    with st.expander("🔎 Validação (pergunta objetiva – uma por vez)", expanded=True):
        if st.button("Gerar pergunta objetiva de validação"):
            with st.spinner("Gerando pergunta objetiva..."):
                pergunta = chain_validar(modelo, st.session_state.requisitos_texto)
                st.session_state.ultima_pergunta = pergunta.strip()
            st.write(f"**Pergunta:** {st.session_state.ultima_pergunta}")

        if st.session_state.ultima_pergunta:
            resposta_val = st.text_input("Sua resposta (objetiva) à pergunta acima")
            if st.button("Aplicar refinamento nos requisitos"):
                if not resposta_val.strip():
                    st.warning("Responda a pergunta antes de refinar.")
                else:
                    with st.spinner("Refinando requisitos..."):
                        atualizados = chain_refinar(
                            modelo,
                            st.session_state.requisitos_texto,
                            st.session_state.ultima_pergunta,
                            resposta_val
                        )
                        # SUBSTITUI (não anexar) — especificação completa consolidada
                        st.session_state.requisitos_texto = atualizados.strip()
                        st.session_state.ultima_pergunta = ""
                    st.success("Requisitos atualizados")
                    st.code(st.session_state.requisitos_texto, language="markdown")

# -------- Conflitos --------
st.markdown("Detecção de Conflitos entre Stakeholders")
if st.button("Analisar conflitos"):
    if not st.session_state.requisitos_texto.strip():
        st.warning("Gere requisitos primeiro.")
    else:
        with st.spinner("Analisando conflitos..."):
            conflitos = chain_conflitos(modelo, st.session_state.requisitos_texto)
        st.text_area("Relatório de Conflitos", value=conflitos, height=240)

# -------- Exportação PDF --------
if st.session_state.requisitos_texto:
    if st.button("Exportar requisitos"):
        caminho = exportar_requisitos_pdf(st.session_state.requisitos_texto)
        with open(caminho, "rb") as f:
            st.download_button("Baixar requisitos.pdf", f, file_name="requisitos_iso29148.pdf", mime="application/pdf")

# -------- Histórico --------
with st.expander("Histórico de Mensagens"):
    for role, text in st.session_state.historico:
        st.chat_message("user" if role == "user" else "assistant").write(text)
