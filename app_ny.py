import numpy as np
from scipy.stats import norm
import streamlit as st
import time
import pandas as pd
import random
import copy 
import threading 

# ... (Klasse Student og alle funksjoner: initialize_state, get_global_state, 
# update_stock_price, black_scholes_price, submit_limit_order, process_market_order, 
# display_order_book, etc. BEHOLDES FRA V13.0) ... 

# ==============================================================================
# 0. GLOBAL MARKEDSTILSTAND (Felles for alle spillere)
# ==============================================================================
if 'GLOBAL_STATE_CONTAINER' not in st.session_state:
    st.session_state.GLOBAL_STATE_CONTAINER = {
        'market_params': {
            'ticker': 'OP_C100',
            'S': 105.0,        
            'K': 100.0,       
            't': 0.25,        
            'r': 0.04,        
            'sigma': 0.30     
        },
        'call_bids': [],
        'call_asks': [],
        'last_update_time': time.time(),
        'simulation_active': False,
        'order_counter': 0,
        'trade_counter': 0
    }

GLOBAL_STATE_LOCK = threading.Lock() 

# Kjernefunksjonene må inkluderes her, men jeg utelater dem for korthet.
# De må kopieres inn fra V13.0.

# ==============================================================================
# 3. HOVED APPLIKASJONSFLYT (MED AUTOMATISK PÅLOGGING)
# ==============================================================================

def main():
    st.set_page_config(page_title="Opsjonsmarked Simulator V14 (Auto-Login)", layout="wide")
    
    # Fjern HTML auto-refresh koden fra V13.0!
    
    initialize_state()

    # --- NY PÅLOGGINGS-/SYNCH-LOGIKK ---
    
    # 1. Prøv å logge inn via Query Parameters i URL-en
    query_params = st.query_params
    
    if st.session_state.active_student is None:
        user_id_from_url = query_params.get("user_id", [None])[0]
        
        # Sjekk om vi kan logge inn automatisk
        if user_id_from_url in st.session_state.students:
            st.session_state.active_student = st.session_state.students[user_id_from_url]
            st.session_state.user_role = st.session_state.active_student.role
            # IKKE st.rerun() her, bare fortsett renderingen
        
        # 2. Vis påloggingsskjema hvis ikke logget inn
        if st.session_state.active_student is None:
            st.title("Opsjonsmarked Simulator: Logg Inn")
            with st.form("login_form"):
                student_id = st.text_input("Student ID/Navn", value="StudentA101")
                role = st.selectbox("Velg Rolle", ['MAKER (Market Maker)', 'TRADER (Pristaker)'])
                submitted = st.form_submit_button("Start Handel", type="primary")
                if submitted:
                    role_type = role.split(' ')[0]
                    
                    if student_id not in st.session_state.students:
                        initial_cash = 100000.0 if role_type == 'MAKER' else 50000.0
                        st.session_state.students[student_id] = Student(student_id, role_type, initial_cash=initial_cash)
                    
                    # KRITISK: Sett bruker-ID i URL-en
                    st.query_params["user_id"] = student_id
                    
                    st.session_state.active_student = st.session_state.students[student_id]
                    st.session_state.user_role = role_type
                    st.rerun() # Nødvendig for å oppdatere UI med den nye URL-en
            return # Avslutt main() hvis brukeren fortsatt er i påloggingsfasen
    # --- SLUTT PÅ NY PÅLOGGINGS-/SYNCH-LOGIKK ---
    
    # ... (Resten av UI-koden følger) ...
    
    st.title(f"🏛️ Opsjonsmarked Simulator (Rolle: {st.session_state.user_role})")
    
    # Statusmeldinger (Sesjonsspesifikk)
    status_msg_placeholder = st.empty()
    if st.session_state.status_message:
        msg = st.session_state.status_message
        if msg['type'] == 'success': status_msg_placeholder.success(msg['content'])
        else: status_msg_placeholder.error(msg['content'])
        st.session_state.status_message = None 

    # Sidefelt for kontroller
    sidebar_container = st.sidebar.container()
    sidebar_container.title("Kontroller")
    
    global_state = get_global_state()
    market_t = global_state['market_params']['t']
    
    # --- START/STOPP SIMULERING ---
    sim_active = global_state['simulation_active']
    
    if market_t > 0 and not sim_active:
        if sidebar_container.button("▶️ Start Simulering", type="primary"):
            with GLOBAL_STATE_LOCK:
                global_state['simulation_active'] = True
                global_state['last_update_time'] = time.time() # Nullstill tid ved start
            st.rerun() 

    # --- TIMER OG PRISOPPDATERING (STABIL LOGIKK fra V12/V13) ---
    WAIT_SECONDS = 60 # Oppdateringsintervall
    timer_placeholder = sidebar_container.empty()
    
    if sim_active and market_t > 0:
        sidebar_container.button("⏸️ Simulering pågår (Synkronisert)", disabled=True)
        
        last_update = global_state['last_update_time']
        current_time = time.time()
        time_elapsed = current_time - last_update
        
        # 1. Oppdater Global State hvis tiden er ute
        if time_elapsed >= WAIT_SECONDS:
            update_stock_price()
            timer_placeholder.success(f"Automatisk prisoppdatering! Ny pris: ${global_state['market_params']['S']:.2f}")
            st.rerun() # Tvinger en umiddelbar rerun etter prisendring for rask tilbakemelding
        
        # 2. Nedtelling
        time_remaining = max(0, WAIT_SECONDS - time_elapsed)
        timer_placeholder.info(f"Pris oppdatering om {int(time_remaining) + 1} sekunder...")
        
    elif market_t <= 0:
        sidebar_container.error("Opsjonen har utløpt.")

    # --- UI RENDERING ---
    tab1, tab2, tab3 = st.tabs(["Markedsinnsikt", "Handelsplass", "Portefølje"])

    with tab1: display_market_info()
    with tab2: trading_interface()
    with tab3: display_portfolio()

if __name__ == '__main__':
    main()
