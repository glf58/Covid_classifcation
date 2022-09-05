from collections import OrderedDict
import streamlit as st
import config

from tabs import intro, EDA_tab, jouons, premier_mod, transfert_learning, poumons, generate_masks, conclusion


st.set_page_config(
    page_title=config.TITLE,
    page_icon="https://datascientest.com/wp-content/uploads/2020/03/cropped-favicon-datascientest-1-32x32.png",
)

path = "C:/Users/guillaume/Documents/Projet_Covid/Streamlit/streamlit_app/"

with open(path+"style.css", "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


TABS = OrderedDict(
    [
        (intro.sidebar_name, intro),
        (EDA_tab.sidebar_name, EDA_tab),
        (premier_mod.sidebar_name, premier_mod),
        (transfert_learning.sidebar_name, transfert_learning),
        (poumons.sidebar_name, poumons),
        (generate_masks.sidebar_name, generate_masks),
        (jouons.sidebar_name, jouons), 
        (conclusion.sidebar_name, conclusion)
    ]
)


def run():
    st.sidebar.image(
        "https://dst-studio-template.s3.eu-west-3.amazonaws.com/logo-datascientest.png",
        width=200,
    )
    tab_name = st.sidebar.radio("", list(TABS.keys()))
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {config.PROMOTION}")

    st.sidebar.markdown("### Team members:")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    tab = TABS[tab_name]

    tab.run()


if __name__ == "__main__":
    run()
