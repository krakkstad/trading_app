import numpy as np
from scipy.stats import norm
import streamlit as st
import time
import pandas as pd
import random
import threading 
import copy 

# ==============================================================================
# 0. GLOBAL MARKEDSTILSTAND (Felles for alle spillere)
# ==============================================================================

# VIKTIG: Dette er den globale tilstanden. Den lagres i selve appens minne.
# Alle sesjoner MÅ lese og skrive til denne.
if 'global_market_state' not in st.session_state:
    st.session_state.global_market_state = {
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
        'status_message': None,
        'order_counter': 0,
        'trade_counter': 0
    }

GLOBAL_STATE_LOCK = threading.Lock() # For å unngå race conditions når vi oppdaterer GLOBAL_STATE


# ==============================================================================
# 1. INITIALISERING OG KJERNEFUNKSJONER (Oppdatert for Global State)
# ==============================================================================

class Student:
    """Enkel klasse for å holde Portefølje- og Kontantdata (Sesjonsspesifikk)."""
    def __init__(self, student_id, role, initial_cash=100000.0):
        self.id = student_id
        self.role = role  
        self.cash = initial_cash
        self.portfolio = {'OP_C100': 0} 
        self.transactions = [] 

def initialize_state():
    """Initialiserer Streamlit session state (kun student-spesifikk data)."""
    if 'initialized_session' not in st.session_state:
        st.session_state.students = {}
        st.session_state.active_student = None
        st.session_state.user_role = None
        st.session_state.initialized_session = True


def update_stock_price():
    """Simulerer aksjekurs og oppdaterer GLOBAL_MARKET_STATE."""
    with GLOBAL_STATE_LOCK:
        market = st.session_state.global_market_state['market_params']
        S, r, sigma = market['S'], market['r'], market['sigma']
        last_time = st.session_state.global_market_state['last_update_time']
        
        time_elapsed = time.time() - last_time
        delta_t_years = 1.0 / 525600.0 * time_elapsed 
        
        Z = np.random.standard_normal()
        dS = S * (r * delta_t_years + sigma * np.sqrt(delta_t_years) * Z)
        
        st.session_state.global_market_state['market_params']['S'] = max(0.01, S + dS) 
        st.session_state.global_market_state['market_params']['t'] = max(0, market['t'] - delta_t_years)
        st.session_state.global_market_state['last_update_time'] = time.time()


def submit_limit_order(student_id, option_type, side, price, quantity):
    """Legger inn en ordre i den globale ordreboken."""
    with GLOBAL_STATE_LOCK:
        if option_type != 'CALL':
            return False, "Kun CALL-opsjoner støttes."

        state = st.session_state.global_market_state
        book_key = 'call_bids' if side == 'BID' else 'call_asks'
        
        state['order_counter'] += 1
        order_id = f"ORD_{state['order_counter']}"
        
        order = {
            'order_id': order_id, 
            'id': student_id,
            'price': price,
            'quantity': quantity,
            'side': side,
            'time': time.time()
        }
        
        state[book_key].append(order)
        
        if side == 'BID':
            state[book_key].sort(key=lambda x: x['price'], reverse=True)
            return True, f"Limitordre lagt inn: BUD {quantity} @ {price:.2f} (ID: {order_id})"
        else: 
            state[book_key].sort(key=lambda x: x['price'])
            return True, f"Limitordre lagt inn: TILBUD {quantity} @ {price:.2f} (ID: {order_id})"


def cancel_order_by_id(order_id, student_id):
    """Fjerner en ordre fra ordreboken basert på ID."""
    with GLOBAL_STATE_LOCK:
        state = st.session_state.global_market_state
        keys = ['call_bids', 'call_asks']
        found = False
        
        for key in keys:
            new_list = []
            for order in state[key]:
                if order.get('order_id') == order_id and order.get('id') == student_id:
                    found = True
                else:
                    new_list.append(order)
            
            if found:
                state[key] = new_list
                if key == 'call_bids':
                    state[key].sort(key=lambda x: x['price'], reverse=True)
                else:
                    state[key].sort(key=lambda x: x['price'])
                return True
                
        return False

# ... (black_scholes_price, black_scholes_greeks forbli de samme)
def black_scholes_price(S, K, t, r, sigma, option_type='call'):
    if t <= 0: return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def black_scholes_greeks(S, K, t, r, sigma, option_type='call'):
    if t <= 0: return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    N_prime_d1 = norm.pdf(d1)
    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    gamma = N_prime_d1 / (S * sigma * np.sqrt(t))
    vega = S * N_prime_d1 * np.sqrt(t) * 0.01 
    return {'delta': delta, 'gamma': gamma, 'theta': 0.0, 'vega': vega}


def process_market_order(taker_id, side, quantity_remaining):
    """
    Gjennomfører en Market Order mot Limit Ordrer i den globale boken.
    """
    with GLOBAL_STATE_LOCK:
        state = st.session_state.global_market_state
        taker = st.session_state.students[taker_id]

        if side == 'BUY':
            book_key = 'call_asks' 
        else: # SELL
            book_key = 'call_bids' 

        limit_book = state[book_key]
        temp_book = copy.deepcopy(limit_book) 
        new_limit_book = []
        
        filled_quantity = 0
        total_cost = 0

        # ... (Matchingslogikken er beholdt, men den opererer på den globale state) ...
        for limit_order in temp_book:
            if quantity_remaining <= 0:
                new_limit_book.append(limit_order) 
                continue

            # Må hente Makerens data fra student state (som er unik for hver sesjon)
            maker = st.session_state.students.get(limit_order['id'])
            if not maker:
                new_limit_book.append(limit_order)
                continue 

            qty_to_trade = min(quantity_remaining, limit_order['quantity'])
            trade_price = limit_order['price']
            trade_amount = qty_to_trade * trade_price

            # Sjekk taker sin kjøpekraft/posisjon FØR handelen
            if side == 'BUY' and taker.cash < trade_amount:
                new_limit_book.append(limit_order) 
                break 
            if side == 'SELL' and taker.portfolio.get('OP_C100', 0) < quantity_remaining:
                return False, "Mangler nok opsjoner å selge!"

            # Oppdatering av Market Taker (Trader)
            if side == 'BUY':
                taker.cash -= trade_amount
                taker.portfolio['OP_C100'] = taker.portfolio.get('OP_C100', 0) + qty_to_trade
            else: # SELL
                taker.cash += trade_amount
                taker.portfolio['OP_C100'] = taker.portfolio.get('OP_C100', 0) - qty_to_trade

            # Oppdatering av Limit Maker (Market Maker)
            if side == 'BUY': # Limit Maker solgte til Trader
                maker.cash += trade_amount
                maker.portfolio['OP_C100'] = maker.portfolio.get('OP_C100', 0) - qty_to_trade
            else: # Limit Maker kjøpte fra Trader
                maker.cash -= trade_amount
                maker.portfolio['OP_C100'] = maker.portfolio.get('OP_C100', 0) + qty_to_trade

            # Oppdater Market og Trade-logg
            quantity_remaining -= qty_to_trade
            filled_quantity += qty_to_trade
            total_cost += trade_amount
            
            state['trade_counter'] += 1
            trade_id = f"TRD_{state['trade_counter']}"

            transaction_record = {
                'id': trade_id,
                'taker': taker_id,
                'maker': maker.id,
                'quantity': qty_to_trade,
                'price': trade_price,
                'time': time.time()
            }
            taker.transactions.append(transaction_record)
            maker.transactions.append(transaction_record) 
            
            # Oppdater resten av Limit Orderen
            if qty_to_trade < limit_order['quantity']:
                limit_order['quantity'] -= qty_to_trade
                new_limit_book.append(limit_order) 
        
        # Sett den nye, oppdaterte boken tilbake i Global State
        state[book_key] = new_limit_book
        
        # Sortering etter oppdatering
        if side == 'BUY':
            state[book_key].sort(key=lambda x: x['price']) 
        else: 
            state[book_key].sort(key=lambda x: x['price'], reverse=True) 

        if filled_quantity > 0:
            avg_price = total_cost / filled_quantity
            return True, f"Market Order utført: {filled_quantity} opsjoner @ {avg_price:.2f} i snitt. Gjenstår: {quantity_remaining}"
        elif filled_quantity == 0 and quantity_remaining > 0:
            return False, f"Market Order feilet: Ordreboken er tom eller ingen tilgjengelige priser."
        else:
            return False, f"Market Order feilet: Ukjent feil."


# ==============================================================================
# 3. GLOBAL MARKEDSTRÅD (Utenfor Streamlit rendering)
# ==============================================================================

def market_update_thread_function():
    """
    Funksjonen som kjører i bakgrunnen og oppdaterer markedet hvert 60. sekund.
    """
    WAIT_SECONDS = 60
    
    while True:
        with GLOBAL_STATE_LOCK:
            if st.session_state.global_market_state['simulation_active'] and st.session_state.global_market_state['market_params']['t'] > 0:
                
                # Sjekker om tiden er ute
                time_elapsed = time.time() - st.session_state.global_market_state['last_update_time']
                
                if time_elapsed >= WAIT_SECONDS:
                    update_stock_price()
                    
                    # Sett statusmelding og flagg for rerunn for ALLE aktive økter
                    st.session_state.global_market_state['status_message'] = {
                        'type': 'success', 
                        'content': f"Automatisk prisoppdatering! Ny pris: ${st.session_state.global_market_state['market_params']['S']:.2f}"
                    }
                    
                    # VIKTIG: Kaller Streamlit sin innebygde funksjon for å tvinge alle aktive
                    # økter til å oppdatere seg. Dette er løsningen for multi-bruker sync.
                    st.rerun() # Denne kommandoen vil nå trigge alle aktive sesjoner
                    
        time.sleep(1) # Sover i 1 sekund for å spare CPU

# Start tråden kun én gang
if 'market_thread' not in st.session_state:
    st.session_state.market_thread = threading.Thread(target=market_update_thread_function, daemon=True)
    st.session_state.market_thread.start()


# ==============================================================================
# 4. UI-KOMPONENTER (Oppdatert for Global State)
# ==============================================================================

def display_order_book():
    """Viser ordreboken fra den globale tilstanden."""
    st.subheader("📚 Ordrebok Status (CALL)")
    state = st.session_state.global_market_state
    
    bids = state['call_bids']
    asks = state['call_asks']
    
    # ... (Visningslogikk) ...
    df_bids = pd.DataFrame(bids).drop(columns=['time', 'side']) if bids else pd.DataFrame(columns=['order_id', 'id', 'price', 'quantity'])
    df_asks = pd.DataFrame(asks).drop(columns=['time', 'side']) if asks else pd.DataFrame(columns=['order_id', 'id', 'price', 'quantity'])
    
    col_books = st.columns(2)
    with col_books[0]:
        st.caption("Bud (BIDS - Kjøpere)")
        st.dataframe(df_bids.head(5).rename(columns={'id': 'Maker ID'}), use_container_width=True)
    with col_books[1]:
        st.caption("Tilbud (ASKS - Selgere)")
        st.dataframe(df_asks.head(5).rename(columns={'id': 'Maker ID'}), use_container_width=True)


def display_market_info():
    """Viser pris og Greeks fra den globale tilstanden."""
    market = st.session_state.global_market_state['market_params']
    S, K, t, r, sigma = market['S'], market['K'], market['t'], market['r'], market['sigma']
    
    st.subheader("⚙️ Nåværende Markedsforhold")
    
    col_params = st.columns(3)
    col_params[0].metric("Underliggende Pris (S)", f"${S:.2f}")
    col_params[1].metric("Innfrielseskurs (K)", f"${K:.2f}")
    col_params[2].metric("Tid til Utløp (t)", f"{t:.4f} år")
    
    call_price = black_scholes_price(S, K, t, r, sigma, 'call')
    put_price = black_scholes_price(S, K, t, r, sigma, 'put')
    call_greeks = black_scholes_greeks(S, K, t, r, sigma, 'call')
    
    st.subheader("💰 Black-Scholes Fair Value")
    col_price = st.columns(2)
    col_price[0].metric("Call Opsjon", f"${call_price:.4f}")
    col_price[1].metric("Put Opsjon", f"${put_price:.4f}")

    st.subheader("📐 Opsjonsgrekere (Call)")
    g_cols = st.columns(4)
    g_cols[0].metric("Delta (Δ)", f"{call_greeks['delta']:.4f}")
    g_cols[1].metric("Gamma (Γ)", f"{call_greeks['gamma']:.4f}")
    g_cols[2].metric("Theta (Θ)", f"{call_greeks['theta']:.4f}")
    g_cols[3].metric("Vega (ν)", f"{call_greeks['vega']:.4f}")

    st.markdown("---")
    display_order_book()


# ... (trading_interface og display_portfolio er beholdt, men bruker global state for ordrer/matching)
def trading_interface():
    active_student = st.session_state.active_student
    
    if active_student.role == 'MAKER':
        st.subheader("✍️ Legg Inn Limit Order (Market Maker)")
        with st.form("maker_form"):
            col_type = st.columns(2)
            option_type = col_type[0].selectbox("Opsjonstype", ['CALL'])
            side = col_type[1].selectbox("Ordre Side", ['BID (Kjøp)', 'ASK (Salg)'])
            col_order = st.columns(2)
            price = col_order[0].number_input("Limit Pris per Kontrakt", min_value=0.01, format="%.2f", key="maker_price")
            quantity = col_order[1].number_input("Antall Kontrakter", min_value=1, step=1, key="maker_qty")
            submitted = st.form_submit_button("Send Limit Order", type="primary")
            
            if submitted:
                side_key = side.split(' ')[0]
                success, msg = submit_limit_order(active_student.id, option_type, side_key, price, quantity)
                if success:
                    st.session_state.global_market_state['status_message'] = {'type': 'success', 'content': msg}
                else:
                    st.session_state.global_market_state['status_message'] = {'type': 'error', 'content': msg}
                st.rerun() 
    
    elif active_student.role == 'TRADER':
        st.subheader("💸 Send Market Order (Trader)")
        st.info(f"Din kontantsaldo: ${active_student.cash:.2f} | Opsjoner: {active_student.portfolio.get('OP_C100', 0)}")
        with st.form("trader_form"):
            col_type = st.columns(2)
            side = col_type[0].selectbox("Ordre Side", ['BUY', 'SELL'])
            quantity = col_type[1].number_input("Antall Kontrakter", min_value=1, step=1, key="trader_qty")
            submitted = st.form_submit_button(f"Send Market {side}", type="primary")
            
            if submitted:
                success, msg = process_market_order(active_student.id, side, quantity)
                if success:
                    st.session_state.global_market_state['status_message'] = {'type': 'success', 'content': msg}
                else:
                    st.session_state.global_market_state['status_message'] = {'type': 'error', 'content': msg}
                st.rerun() 

def display_portfolio():
    student = st.session_state.students[st.session_state.active_student.id]
    st.subheader(f"👤 Porteføljeoppsummering for {student.id} (Rolle: {student.role})")
    col_metrics = st.columns(2)
    col_metrics[0].metric("Tilgjengelig Kontantbeholdning", f"${student.cash:.2f}")
    col_metrics[1].metric("Opsjonsbeholdning (OP_C100)", f"{student.portfolio.get('OP_C100', 0)}")

    st.subheader("📋 Transaksjonslogg")
    if student.transactions:
        df_trades = pd.DataFrame(student.transactions)
        st.dataframe(df_trades, use_container_width=True)
    else:
        st.info("Ingen gjennomførte handler.")
    
    st.subheader("🛑 Åpne Limitordrer (Kansellering)")
    # Henter fra global state
    open_bids = [o for o in st.session_state.global_market_state['call_bids'] if o['id'] == student.id]
    open_asks = [o for o in st.session_state.global_market_state['call_asks'] if o['id'] == student.id]
    open_orders = open_bids + open_asks
    
    if open_orders:
        df_orders = pd.DataFrame(open_orders).drop(columns=['id', 'time'])
        df_orders = df_orders.rename(columns={'order_id': 'ID', 'price': 'Pris', 'quantity': 'Antall', 'side': 'Side'})
        st.dataframe(df_orders, use_container_width=True)
        
        with st.form("cancel_form"):
            order_to_cancel = st.selectbox("Velg Ordre ID for kansellering", df_orders['ID'].tolist())
            cancel_submitted = st.form_submit_button("❌ Kanseller Ordre", type="secondary")
            
            if cancel_submitted:
                success = cancel_order_by_id(order_to_cancel, student.id)
                if success:
                    st.session_state.global_market_state['status_message'] = {'type': 'success', 'content': f"Ordre {order_to_cancel} kansellert."}
                else:
                    st.session_state.global_market_state['status_message'] = {'type': 'error', 'content': f"Kansellering feilet for ID {order_to_cancel}. (Feil eier?)"}
                st.rerun() 
    else:
        st.info("Ingen åpne limitordrer.")


# ==============================================================================
# 5. HOVED APPLIKASJONSFLYT (Fjerner lokal while-løkke)
# ==============================================================================

def main():
    st.set_page_config(page_title="Opsjonsmarked Simulator V9 (Multi-User Sync)", layout="wide")
    
    initialize_state()

    # --- LOGIN ---
    if st.session_state.active_student is None:
        # ... (Login logikk, som også oppretter studenten i st.session_state.students) ...
        st.title("Opsjonsmarked Simulator: Logg Inn")
        with st.form("login_form"):
            student_id = st.text_input("Student ID/Navn", value="StudentA101")
            role = st.selectbox("Velg Rolle", ['MAKER (Market Maker)', 'TRADER (Pristaker)'])
            submitted = st.form_submit_button("Start Handel", type="primary")
            if submitted:
                role_type = role.split(' ')[0]
                
                # Sjekk om studenten allerede eksisterer i NOEN sesjon (for å gjenopprette state)
                if student_id not in st.session_state.students:
                    initial_cash = 100000.0 if role_type == 'MAKER' else 50000.0
                    st.session_state.students[student_id] = Student(student_id, role_type, initial_cash=initial_cash)
                
                st.session_state.active_student = st.session_state.students[student_id]
                st.session_state.user_role = role_type
                st.rerun() 
        return

    st.title(f"🏛️ Opsjonsmarked Simulator (Rolle: {st.session_state.user_role})")
    
    # Statusmeldinger (Henter fra global state for å synkronisere)
    status_msg_placeholder = st.empty()
    if st.session_state.global_market_state['status_message']:
        msg = st.session_state.global_market_state['status_message']
        if msg['type'] == 'success':
            status_msg_placeholder.success(msg['content'])
        else:
            status_msg_placeholder.error(msg['content'])
        # Viser meldingen kun én gang i hver sesjon:
        st.session_state.global_market_state['status_message'] = None 

    # Sidefelt for kontroller
    sidebar_container = st.sidebar.container()
    sidebar_container.title("Kontroller")

    market_t = st.session_state.global_market_state['market_params']['t']
    
    # --- START/STOPP SIMULERING ---
    sim_active = st.session_state.global_market_state['simulation_active']
    
    if market_t > 0 and not sim_active:
        if sidebar_container.button("▶️ Start Simulering", type="primary"):
            with GLOBAL_STATE_LOCK:
                st.session_state.global_market_state['simulation_active'] = True
            st.rerun() 
    
    # --- TIMER OG SYNCH STATUS ---
    if sim_active and market_t > 0:
        sidebar_container.button("⏸️ Simulering pågår (Globalt Synkronisert)", disabled=True)
        
        # Henter tid fra global state for å vise tid til neste oppdatering
        time_elapsed = time.time() - st.session_state.global_market_state['last_update_time']
        time_remaining = max(0, 60 - time_elapsed)
        
        # Viser tid uten å bruke en while-løkke/time.sleep (Streamlit håndterer dette)
        sidebar_container.info(f"Pris oppdatering om {int(time_remaining) + 1} sekunder...")
    
    elif market_t <= 0:
        sidebar_container.error("Opsjonen har utløpt.")

    # --- UI RENDERING ---
    tab1, tab2, tab3 = st.tabs(["Markedsinnsikt", "Handelsplass", "Portefølje"])

    with tab1:
        display_market_info()

    with tab2:
        trading_interface()

    with tab3:
        display_portfolio()


if __name__ == '__main__':
    main()
