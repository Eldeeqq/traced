## implementation of TOC (Table of Contents) for streamlit
# https://discuss.streamlit.io/t/table-of-contents-widget/3470/8
# author: https://discuss.streamlit.io/u/okld

import streamlit as st


class Toc:
    def __init__(self):
        self._items = []
        self._placeholder = None

    def title(self, text):
        self._markdown(text, "h1")

    def header(self, text):
        self._markdown(text, "h2", " " * 2)

    def subheader(self, text):
        self._markdown(text, "h3", " " * 4)

    def placeholder(self, sidebar=False):
        self._placeholder = (
            st.sidebar.expander("Table of contents")
            if sidebar
            else st.expander("Table of contents")
        )

    def generate(self):
        if self._placeholder:
            self._placeholder.markdown("\n".join(self._items), unsafe_allow_html=True)

    def _markdown(self, text, level, space=""):
        key = "-".join(text.split()).lower()

        st.markdown(f"<{level} id='{key}'>{text}</{level}>", unsafe_allow_html=True)
        self._items.append(f"{space}* <a href='#{key}'>{text}</a>")

    def reset(self):
        self._items.clear()


TOC = Toc()
