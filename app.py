import os
import json
import itertools
import joblib
import requests
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import gradio as gr

model_to_use = joblib.load("ddi_model.joblib")

with open("drug_graph.json", "r") as f:
    graph_data = json.load(f)

G = nx.Graph()
G.add_nodes_from(graph_data["drugs"])
G.add_edges_from(graph_data["edges"])
all_drugs = sorted(list(G.nodes()))

KNOWN_REASONS = {
    frozenset(["Warfarin", "Aspirin"]): "Additive anticoagulant and antiplatelet effects significantly increase the risk of severe gastrointestinal or internal bleeding.",
    frozenset(["Warfarin", "Ibuprofen"]): "Ibuprofen inhibits platelet function and can irritate the stomach lining while displacing Warfarin, elevating major bleeding risk.",
    frozenset(["Aspirin", "Ibuprofen"]): "NSAIDs like Ibuprofen can compete with Aspirin's antiplatelet binding site and jointly increase stomach ulcer and bleeding risk.",
    frozenset(["Amoxicillin", "Acetaminophen"]): "No major direct chemical interaction is typically reported; generally considered low risk at standard therapeutic doses."
}

def get_clinical_reason(drug_1, drug_2, risk_level, is_confirmed, shared_neighbors):
    pair_key = frozenset([drug_1, drug_2])

    groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if groq_api_key:
        try:
            prompt = (
                f"In 2 concise, plain-English clinical sentences, explain the pharmacological reason why taking "
                f"{drug_1} and {drug_2} together may cause a drug-drug interaction (Risk Level: {risk_level}). "
                f"Focus on the biological mechanism (e.g., liver enzyme CYP450 competition, bleeding risk, blood pressure, "
                f"organ toxicity, or reduced efficacy). If they are generally safe together, state that clearly."
            )
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 110
                },
                timeout=8
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            pass

    if pair_key in KNOWN_REASONS:
        return KNOWN_REASONS[pair_key]

    if shared_neighbors:
        sample_shared = ", ".join(shared_neighbors[:3])
        if is_confirmed:
            return f"Confirmed adverse interaction in DrugBank. Both drugs share {len(shared_neighbors)} mutual interacting drugs (including {sample_shared}), indicating overlapping metabolic or physiological pathways."
        elif risk_level != "Low Risk":
            return f"High network similarity detected: both drugs share {len(shared_neighbors)} common interacting neighbors (e.g., {sample_shared}), suggesting potential metabolic competition or additive side effects."
    return "Minimal overlapping interaction pathways detected in the drug network; low likelihood of an adverse chemical interaction."


def build_network_plot(selected_drugs, pair_results):
    sub_G = nx.Graph()
    for d in selected_drugs:
        sub_G.add_node(d, node_type="selected")

    for res in pair_results:
        u, v = res["Drug 1"], res["Drug 2"]
        sub_G.add_edge(u, v)
        for neighbor in res["Shared_List"][:3]:
            sub_G.add_node(neighbor, node_type="shared")
            sub_G.add_edge(u, neighbor)
            sub_G.add_edge(v, neighbor)

    pos = nx.spring_layout(sub_G, seed=42, k=0.8)

    edge_x, edge_y = [], []
    for u, v in sub_G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color="#94a3b8"),
        hoverinfo="none", mode="lines"
    )

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in sub_G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        if sub_G.nodes[node]["node_type"] == "selected":
            node_color.append("#e11d48")
            node_size.append(24)
        else:
            node_color.append("#3b82f6")
            node_size.append(12)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=node_text, textposition="top center",
        hoverinfo="text",
        marker=dict(color=node_color, size=node_size, line=dict(width=2, color="white"))
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Interactive Drug Interaction Sub-Network (Red = Selected | Blue = Shared Neighbors)",
        showlegend=False, hovermode="closest",
        margin=dict(b=20, l=20, r=20, t=45),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#f8fafc", paper_bgcolor="white", height=420
    )
    return fig


def analyze_prescription(selected_drugs):
    if not selected_drugs or len(selected_drugs) < 2:
        return (
            "<div style='padding:16px; border-radius:10px; background:#fef3c7; color:#92400e; font-weight:600;'>"
            "Please select at least 2 drugs to analyze polypharmacy risk.</div>",
            "",
            pd.DataFrame(),
            None
        )

    pairs = list(itertools.combinations(selected_drugs, 2))
    pair_results = []
    reason_cards_html = "<div style='display:flex; flex-direction:column; gap:10px;'>"

    for u, v in pairs:
        cn_list = list(nx.common_neighbors(G, u, v))
        cn = len(cn_list)
        jc = list(nx.jaccard_coefficient(G, [(u, v)]))[0][2]
        aa = list(nx.adamic_adar_index(G, [(u, v)]))[0][2]
        ra = list(nx.resource_allocation_index(G, [(u, v)]))[0][2]
        pa = list(nx.preferential_attachment(G, [(u, v)]))[0][2]
        dd = abs(G.degree(u) - G.degree(v))

        X_pair = pd.DataFrame([{
            "common_neighbors": cn,
            "jaccard": jc,
            "adamic_adar": aa,
            "resource_allocation": ra,
            "preferential_attachment": pa,
            "degree_diff": dd
        }])

        risk_prob = round(float(model_to_use.predict_proba(X_pair)[0][1]) * 100, 1)
        is_confirmed = G.has_edge(u, v)

        if risk_prob > 70 or is_confirmed:
            level, badge_bg, badge_txt = "High Risk", "#ffe4e6", "#be123c"
        elif risk_prob >= 30:
            level, badge_bg, badge_txt = "Moderate Risk", "#fef3c7", "#b45309"
        else:
            level, badge_bg, badge_txt = "Low Risk", "#d1fae5", "#047857"

        reason = get_clinical_reason(u, v, level, is_confirmed, cn_list)
        status_label = "Confirmed in DrugBank" if is_confirmed else "AI Predicted (Untested)"

        reason_cards_html += f"""
        <div style="padding:14px 16px; border-radius:10px; background:#ffffff; border:1px solid #e2e8f0; box-shadow:0 1px 2px rgba(0,0,0,0.04);">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; flex-wrap:wrap; gap:8px;">
                <span style="font-weight:700; font-size:15px; color:#0f172a;">{u} + {v}</span>
                <span style="padding:3px 10px; border-radius:999px; font-size:12px; font-weight:600; background:{badge_bg}; color:{badge_txt};">
                    {level} ({risk_prob}%) · {status_label}
                </span>
            </div>
            <p style="margin:0; font-size:14px; color:#1e293b !important; line-height:1.5;"><b style="color:#000000 !important;">Why:</b> {reason}</p>
        </div>
        """

        pair_results.append({
            "Drug 1": u,
            "Drug 2": v,
            "Risk %": risk_prob,
            "Risk Level": level,
            "Database Status": status_label,
            "Mutual Drugs": cn,
            "Shared_List": cn_list
        })

    reason_cards_html += "</div>"

    max_risk = max(r["Risk %"] for r in pair_results)
    has_high = any(r["Risk Level"] == "High Risk" for r in pair_results)
    has_mod = any(r["Risk Level"] == "Moderate Risk" for r in pair_results)

    if has_high:
        bg, border, text, verdict = "#ffe4e6", "#f43f5e", "#9f1239", "HIGH POLYPHARMACY RISK DETECTED"
    elif has_mod:
        bg, border, text, verdict = "#fef3c7", "#f59e0b", "#92400e", "MODERATE INTERACTION RISK"
    else:
        bg, border, text, verdict = "#d1fae5", "#10b981", "#065f46", "LOW RISK — COMBINATION APPEARS SAFE"

    banner_html = f"""
    <div style="padding:18px; border-radius:12px; background:{bg}; border-left:6px solid {border}; color:{text};">
        <h3 style="margin:0 0 6px 0; font-size:20px;">{verdict}</h3>
        <p style="margin:0; font-size:15px;">Analyzed <b>{len(pairs)}</b> drug pair(s) | Highest Predicted Risk: <b>{max_risk}%</b></p>
    </div>
    """

    display_df = pd.DataFrame(pair_results).drop(columns=["Shared_List"])
    fig = build_network_plot(selected_drugs, pair_results)

    return banner_html, reason_cards_html, display_df, fig


with gr.Blocks(title="DDI Polypharmacy Risk Predictor") as demo:
    gr.Markdown("# Drug-Drug Interaction & Polypharmacy Risk Predictor")
    gr.Markdown("Select **2 to 10 medicines** to predict adverse interaction risks using **NetworkX Graph Topology** + **Random Forest (ROC-AUC: 0.9283)** with **Clinical Mechanism Explanations**.")

    drug_selector = gr.Dropdown(
        choices=all_drugs,
        multiselect=True,
        max_choices=10,
        label="Select Prescribed Medicines (Type to Search)",
        info="Start typing any FDA-approved drug name (e.g., Warfarin, Aspirin, Ibuprofen)"
    )

    analyze_btn = gr.Button("Analyze Prescription Risk", variant="primary", size="lg")

    gr.Examples(
        examples=[
            [["Warfarin", "Aspirin", "Ibuprofen"]],
            [["Amoxicillin", "Acetaminophen"]],
            [["Warfarin", "Acetaminophen", "Amoxicillin"]]
        ],
        inputs=drug_selector,
        label="Quick Sample Prescriptions (Click for Instant Demo)"
    )

    banner_output = gr.HTML()
    reasons_output = gr.HTML(label="Clinical & Pharmacological Explanations")
    table_output = gr.Dataframe(label="Pair-by-Pair Risk Metrics", interactive=False)
    graph_output = gr.Plot(label="Interactive Drug Interaction Network")

    gr.Markdown("*Disclaimer: This application is an academic Data Science prototype for educational purposes only and is not intended as medical advice.*")

    analyze_btn.click(
        fn=analyze_prescription,
        inputs=drug_selector,
        outputs=[banner_output, reasons_output, table_output, graph_output]
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
)
