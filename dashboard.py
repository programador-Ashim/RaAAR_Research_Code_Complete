"""Streamlit display for real CV metrics and explicit synthetic RaAAR demonstration."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from raaar_evidence import gate,load_facts

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='RaAAR Research Demonstration', layout='wide')
st.title('RaAAR — Reliability-Aware Anaphoric Ambiguity Resolution')
st.caption('Research demonstration: real datasets and model experiments + separately labeled synthetic evidence scenarios')
tab1,tab2,tab3=st.tabs(['Evidence verification demo', 'Model experiments', 'Research limits'])
with tab1:
    st.info('These five evidence records and the default linguistic score are synthetic / illustrative. They are not empirical results from the paper.')
    cases=json.loads((ROOT/'evidence/example_scenarios.json').read_text())
    labels={f"{x['scenario'].capitalize()} — {x['id']}":x for x in cases}
    chosen=st.selectbox('Choose an evidence condition',list(labels))
    case=labels[chosen]
    st.write('**Requirement:**',case['context'])
    st.write('**Candidate antecedent:**',case['candidate'],' | **Target system version:**',case['target_version'])
    score=st.slider('Illustrative linguistic confidence',min_value=0.0,max_value=1.0,value=.82,step=.01)
    min_score=st.slider('Evidence gate confidence cutoff',min_value=0.0,max_value=1.0,value=.65,step=.01)
    facts=load_facts(ROOT/'evidence'/case['facts_file'])
    result=gate(case['candidate'],case['capability'],case['target_version'],score,facts,min_score)
    c1,c2,c3=st.columns(3)
    c1.metric('Evidence status',result['evidence']['status'])
    c2.metric('Decision',result['decision'])
    c3.metric('Confidence',f'{score:.2f}')
    st.subheader('Traceable evidence facts')
    if facts:st.dataframe(pd.DataFrame([f.__dict__ for f in facts]),hide_index=True,use_container_width=True)
    else:st.write('No linked current evidence found in the demo fact store.')
    with st.expander('Full audit JSON'):
        st.json(result)
with tab2:
    paths=sorted((ROOT/'cv_outputs').rglob('metrics.json')) if (ROOT/'cv_outputs').exists() else []
    if not paths: st.warning('No locally trained CV results found. Run experiment_cv.py before viewing real experiment metrics.')
    else:
        rows=[]
        for f in paths:
            obj=json.loads(f.read_text())
            for dataset in ('damir_cv','reqeval_external'):
                m=obj.get(dataset,{})
                if m.get('n',0):
                    rows.append({'model':obj['model'],'task':obj['task'],'seed':obj['seed'],
                                 'fold':obj['fold'],'dataset':dataset,'n':m['n'],
                                 'f1':m.get('f1'),'f2':m.get('f2'),
                                 'always_positive_f2':m.get('always_positive_f2'),
                                 'predicted_positive':m.get('predicted_positive'),
                                 'top1_accuracy':m.get('top1_accuracy')})
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        st.caption('Only metrics actually loaded from local training output are displayed. One fold is not a complete CV study.')
with tab3:
    st.markdown('''**Implemented:** real XLSX parsing; matched RoBERTa/SpanBERT training code; candidate ranking;
        structured-fact missing, outdated and contradiction checks; selective review decision; test scripts.

**Not yet established:** experimental superiority over published baselines, semantic retrieval/NLI,
        independently annotated 60–80-case knowledge benchmark, Cohen κ, 50-case human error analysis,
        validated risk–coverage improvements, or full zero-shot LLM evaluation.

        Do not use synthetic status demo results as evidence for scientific claims.''')
