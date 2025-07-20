import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from glob import glob

st.set_page_config(
    page_title="LLM OCR Comparison Dashboard",
    page_icon="📊",
    layout="wide"
)


def load_results():
    results_pattern = "../results/**/benchmark_results_*.json"
    result_files = glob(results_pattern, recursive=True)

    if not result_files:
        return None

    latest_file = max(result_files, key=os.path.getctime)

    with open(latest_file, 'r') as f:
        data = json.load(f)

    return pd.DataFrame(data), latest_file


def main():
    st.title("🔍 LLM OCR Comparison Dashboard")
    st.markdown("---")

    data = load_results()
    if data is None:
        st.error("No benchmark results found! Please run the benchmark first.")
        st.code("python src/main.py")
        return

    df, results_file = data
    st.success(f"Loaded results from: {os.path.basename(results_file)}")

    st.sidebar.header("Filters")

    available_models = df['ocr_model'].unique()
    selected_models = st.sidebar.multiselect(
        "Select Models",
        available_models,
        default=available_models
    )

    filtered_df = df[df['ocr_model'].isin(selected_models)]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_tests = len(filtered_df)
        st.metric("Total Tests", total_tests)

    with col2:
        successful_tests = len(filtered_df[filtered_df['error'].isna()])
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")

    with col3:
        avg_json_accuracy = filtered_df['json_accuracy'].mean()
        if pd.notna(avg_json_accuracy):
            st.metric("Avg JSON Accuracy", f"{avg_json_accuracy:.3f}")
        else:
            st.metric("Avg JSON Accuracy", "N/A")

    with col4:
        avg_text_similarity = filtered_df['text_similarity'].mean()
        if pd.notna(avg_text_similarity):
            st.metric("Avg Text Similarity", f"{avg_text_similarity:.3f}")
        else:
            st.metric("Avg Text Similarity", "N/A")

    st.markdown("## 📈 Performance Metrics")

    if 'json_accuracy' in filtered_df.columns and filtered_df['json_accuracy'].notna().any():
        fig_json = px.box(
            filtered_df.dropna(subset=['json_accuracy']),
            x='ocr_model',
            y='json_accuracy',
            title="JSON Accuracy by Model"
        )
        fig_json.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_json, use_container_width=True)

    if 'text_similarity' in filtered_df.columns and filtered_df['text_similarity'].notna().any():
        fig_text = px.box(
            filtered_df.dropna(subset=['text_similarity']),
            x='ocr_model',
            y='text_similarity',
            title="Text Similarity by Model"
        )
        fig_text.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_text, use_container_width=True)

    st.markdown("## 💰 Cost Analysis")

    cost_data = []
    for _, row in filtered_df.iterrows():
        if pd.notna(row['usage']) and row['usage']:
            try:
                if isinstance(row['usage'], str):
                    usage = json.loads(row['usage'])
                else:
                    usage = row['usage']

                cost_data.append({
                    'model': row['ocr_model'],
                    'total_cost': usage.get('total_cost', 0),
                    'input_tokens': usage.get('input_tokens', 0),
                    'output_tokens': usage.get('output_tokens', 0),
                    'duration': usage.get('duration', 0)
                })
            except:
                continue

    if cost_data:
        cost_df = pd.DataFrame(cost_data)

        avg_cost = cost_df.groupby('model')['total_cost'].mean().reset_index()
        fig_cost = px.bar(avg_cost, x='model', y='total_cost', title="Average Cost per Request by Model")
        fig_cost.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_cost, use_container_width=True)

        avg_duration = cost_df.groupby('model')['duration'].mean().reset_index()
        fig_duration = px.bar(avg_duration, x='model', y='duration', title="Average Duration (seconds) by Model")
        fig_duration.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_duration, use_container_width=True)

    st.markdown("## 📋 Detailed Results")

    display_columns = [
        'file_url', 'ocr_model', 'extraction_model',
        'json_accuracy', 'text_similarity', 'error'
    ]

    available_columns = [col for col in display_columns if col in filtered_df.columns]
    st.dataframe(
        filtered_df[available_columns].round(4),
        use_container_width=True
    )

    st.markdown("## 📥 Download Data")
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="benchmark_results.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()