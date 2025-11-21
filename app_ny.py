import numpy as np
from scipy.stats import norm
import streamlit as st
import time
import pandas as pd
import random
import copy 
import threading 
import json # NYTT: For å lagre/laste til/fra JSON

# ==============================================================================
# 0. GLOBAL MARKEDSTILSTAND OG FIL-LAGRING
# ==============================================================================
# Setter opp GLOBAL_STATE_CONTAINER og den nye lagringsmekanismen

STUDENTS_FILE = "students.json" # Filnavnet for persistent lagring
GLOBAL_STATE_LOCK = threading.Lock() 

class Student:
    """Enkel klasse for å holde Portefølje- og Kontantdata."""
    def __init__(self, student_id, role, cash=100000.0, portfolio=None, transactions=None):
        self.id = student_id
        self.role = role  
        self.cash = cash
        self.portfolio = portfolio if portfolio is not None else {'OP_C100': 0} 
        self.transactions = transactions if transactions is not None else [] 
    
    # NYTT: Metoder for serialisering og deserialisering
    def to_dict(self):
        return self.__dict__
    
    @staticmethod
    def from_dict(data):
        return Student(
            student_id=data['id'], 
            role=data['role'], 
            cash=data['cash'], 
            portfolio=data['portfolio'], 
            transactions=data['transactions']
        )

# NYE FUNKSJONER FOR PERSISTENT LAGRING
def load_global_students():
    """Laster Student-objekter fra JSON-fil."""
    try:
        with open(STUDENTS_FILE, 'r') as f:
            data = json.load(f)
            # Konverterer dict tilbake til Student-objekter
            return {k: Student.from_dict(v) for k, v in data.items()}
    except (FileNotFoundError, json.JSONDecodeError):
        return {} # Returnerer tom dict hvis filen er tom eller ikke finnes

def save_global_students(students_dict):
    """Lagrer Student-objekter til JSON-fil."""
    try:
        # Konverterer Student-objekter til dict før lagring
        data_to_save = {k: v.to_dict() for k, v in students_dict.items()}
        with open(STUDENTS_FILE, 'w') as f:
            json.dump(data_to_save, f, indent=4)
    except Exception as e:
        st.error(f"Feil ved lagring av studentdata: {e}")

# Initialisering av global tilstand
if 'GLOBAL_STATE_CONTAINER' not in st.session_state:
    st.session_state.GLOBAL_STATE_CONTAINER = {
        'market_params': {
            'ticker': 'OP_C100', 'S': 105.0, 'K': 100.0, 't': 0.25, 'r': 0.04, 'sigma': 0.30     
        },
        'call_bids': [],
        'call_asks': [],
        'last_update_time': time.time(),
        'simulation_active': False,
        'order_counter': 0,
        'trade_counter': 0,
        'students': load_global_students() # Laster PERSISTENT DATA ved oppstart
    }

# ==============================================================================
# 1. KJERNEFUNKSJONER (Logikk - oppdatert for global student-henting/lagring)
# ==============================================================================

def initialize_state():
    """Initialiserer Streamlit session state (kun sesjonsspesifikk data)."""
    if 'initialized_session' not in st.session_state:
        st.session_state.active_student_id = None 
        st.session_state.user_role = None
        st.session_state.initialized_session = True
        st.session_state.status_message = None 

def get_global_state():
    return st.session_state.GLOBAL_STATE_CONTAINER

def update_stock_price():
    """Simulerer aksjekurs og oppdaterer GLOBAL_MARKET_STATE under lås."""
    with GLOBAL_STATE_LOCK:
        global_state = get_global_state()
        market = global_state['market_params']
        # ... (resten av funksjonen er uendret) ...
        S, r, sigma = market['S'], market['r'], market['sigma']
        last_time = global_state['last_update_time']
        
        time_elapsed = time.time() - last_time
        delta_t_years = 60.0 / 525600.0 
        
        Z = np.random.standard_normal()
        dS = S * (r * delta_t_years + sigma * np.sqrt(delta_t_years) * Z)
        
        global_state['market_params']['S'] = max(0.01, S + dS) 
        global_state['market_params']['t'] = max(0, market['t'] - delta_t_years)
        global_state['last_update_time'] = time.time()

# submit_limit_order er uendret.
def submit_limit_order(student_id, option_type, side, price, quantity):
    """Legger inn en ordre i den globale ordreboken under lås."""
    with GLOBAL_STATE_LOCK:
        if option_type != 'CALL': return False, "Kun CALL-opsjoner støttes."
        state = get_global_state()
        book_key = 'call_bids' if side == 'BID' else 'call_asks'
        state['order_counter'] += 1
        order_id = f"ORD_{state['order_counter']}"
        order = {'order_id': order_id, 'id': student_id, 'price': price, 'quantity': quantity, 'side': side, 'time': time.time()}
        state[book_key].append(order)
        if side == 'BID':
            state[book_key].sort(key=lambda x: x['price'], reverse=True)
            return True, f"Limitordre lagt inn: BUD {quantity} @ {price:.2f} (ID: {order_id})"
        else: 
            state[book_key].sort(key=lambda x: x['price'])
            return True, f"Limitordre lagt inn: TILBUD {quantity} @ {price:.2f} (ID: {order_id})"

# cancel_order_by_id er uendret.
def cancel_order_by_id(order_id, student_id):
    """Fjerner en ordre fra ordreboken basert på ID under lås."""
    with GLOBAL_STATE_LOCK:
        state = get_global_state()
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
                if key == 'call_bids': state[key].sort(key=lambda x: x['price'], reverse=True)
                else: state[key].sort(key=lambda x: x['price'])
                return True
        return False

# process_market_order er oppdatert for å lagre data.
def process_market_order(taker_id, side, quantity_remaining):
    """Gjennomfører en Market Order mot Limit Ordrer i den globale boken under lås."""
    with GLOBAL_STATE_LOCK:
        global_state = get_global_state()
        global_students = global_state['students'] # Hent global student-dict
        
        taker = global_students.get(taker_id) 
        if not taker: return False, "Taker student not found."

        if side == 'BUY': book_key = 'call_asks' 
        else: book_key = 'call_bids' 

        limit_book = global_state[book_key]
        temp_book = copy.deepcopy(limit_book) 
        new_limit_book = []
        filled_quantity = 0
        total_cost = 0

        for limit_order in temp_book:
            if quantity_remaining <= 0:
                new_limit_book.append(limit_order) 
                continue

            maker_id = limit_order['id']
            maker = global_students.get(maker_id) 
            if not maker:
                new_limit_book.append(limit_order)
                continue 

            qty_to_trade = min(quantity_remaining, limit_order['quantity'])
            trade_price = limit_order['price']
            trade_amount = qty_to_trade * trade_price

            if side == 'BUY' and taker.cash < trade_amount:
                new_limit_book.append(limit_order) 
                break 
            if side == 'SELL' and taker.portfolio.get('OP_C100', 0) < quantity_remaining:
                return False, "Mangler nok opsjoner å selge!"

            # --- Oppdatering av Global Student Data ---
            if side == 'BUY':
                taker.cash -= trade_amount
                taker.portfolio['OP_C100'] = taker.portfolio.get('OP_C100', 0) + qty_to_trade
                maker.cash += trade_amount
                maker.portfolio['OP_C100'] = maker.portfolio.get('OP_C100', 0) - qty_to_trade
            else: # SELL
                taker.cash += trade_amount
                taker.portfolio['OP_C100'] = taker.portfolio.get('OP_C100', 0) - qty_to_trade
                maker.cash -= trade_amount
                maker.portfolio['OP_C100'] = maker.portfolio.get('OP_C100', 0) + qty_to_trade
            # --- Slutt Oppdatering ---

            # Oppdater Trade-logg
            quantity_remaining -= qty_to_trade
            filled_quantity += qty_to_trade
            total_cost += trade_amount
            
            global_state['trade_counter'] += 1
            trade_id = f"TRD_{global_state['trade_counter']}"

            transaction_record = {
                'id': trade_id, 'taker': taker_id, 'maker': maker.id, 'quantity': qty_to_trade, 
                'price': trade_price, 'time': time.time()
            }
            taker.transactions.append(transaction_record)
            maker.transactions.append(transaction_record) 
            
            if qty_to_trade < limit_order['quantity']:
                limit_order['quantity'] -= qty_to_trade
                new_limit_book.append(limit_order) 
        
        global_state[book_key] = new_limit_book
        
        # KRITISK: Lagre oppdatert studentdata til filen etter handel!
        save_global_students(global_students)

        if filled_quantity > 0:
            avg_price = total_cost / filled_quantity
            return True, f"Market Order utført: {filled_quantity} opsjoner @ {avg_price:.2f} i snitt. Gjenstår: {quantity_remaining}"
        elif filled_quantity == 0 and quantity_remaining > 0:
            return False, f"Market Order feilet: Ordreboken er tom eller ingen tilgjengelige priser."
        else:
            return False, f"Market Order feilet: Ukjent feil."

# ... (Kopi av UI-funksjonene fra V15.0: black_scholes_price, black_scholes_greeks, 
# get_active_student, display_order_book, display_market_info, trading_interface, 
# display_portfolio - disse er uendret) ...

# --- Kopi av UI-komponenter og Black-Scholes funksjoner ---
# ...
# For å unngå å gjenta 400 linjer kode, antar jeg at du kopierer disse
# uendrede funksjonene fra V15.0 inn her. Hvis du trenger dem, si fra!
# ...
# Bare 'get_active_student()' endres:

def get_active_student():
    """Henter den aktive studenten fra den globale beholdningen."""
    active_id = st.session_state.active_student_id
    # Merk: henter fra global state, som ble lastet fra filen
    return get_global_state()['students'].get(active_id)

# Kopi av alle de andre UI-funksjonene må settes inn her.

# ==============================================================================
# 3. HOVED APPLIKASJONSFLYT (Automatisk Pålogging med Persistent Data)
# ==============================================================================

def main():
    st.set_page_config(page_title="Opsjonsmarked Simulator V16 (Persistent Data)", layout="wide")
    
    initialize_state()

    global_state = get_global_state()
    global_students = global_state['students']
    
    # --- PÅLOGGINGS- OG SYNCH-LOGIKK ---
    
    if st.session_state.active_student_id is None:
        query_params = st.query_params
        user_id_from_url = query_params.get("user_id", [None])[0]
        
        # 1. Prøv å logge inn automatisk via URL-parameter
        if user_id_from_url in global_students:
            st.session_state.active_student_id = user_id_from_url
            st.session_state.user_role = global_students[user_id_from_url].role
        
        # 2. Vis påloggingsskjema hvis ikke logget inn
        if st.session_state.active_student_id is None:
            st.title("Opsjonsmarked Simulator: Logg Inn")
            with st.form("login_form"):
                student_id = st.text_input("Student ID/Navn", value="StudentA101")
                role = st.selectbox("Velg Rolle", ['MAKER (Market Maker)', 'TRADER (Pristaker)'])
                submitted = st.form_submit_button("Start Handel", type="primary")
                if submitted:
                    role_type = role.split(' ')[0]
                    
                    with GLOBAL_STATE_LOCK: # Sikker opprettelse/henting
                        if student_id not in global_students:
                            initial_cash = 100000.0 if role_type == 'MAKER' else 50000.0
                            global_students[student_id] = Student(student_id, role_type, initial_cash=initial_cash)
                        
                        # KRITISK: Lagre den nye studenten til filen!
                        save_global_students(global_students) 
                    
                    # Sett ID lokalt og i URL
                    st.query_params["user_id"] = student_id
                    st.session_state.active_student_id = student_id
                    st.session_state.user_role = role_type
                    st.rerun() 
            return 

    # --- MAIN UI STARTER HER ---
    
    st.title(f"🏛️ Opsjonsmarked Simulator (Rolle: {st.session_state.user_role})")
    
    # ... (Resten av main() er uendret) ...
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
    
    market_t = global_state['market_params']['t']
    
    # --- START/STOPP SIMULERING ---
    sim_active = global_state['simulation_active']
    
    if market_t > 0 and not sim_active:
        if sidebar_container.button("▶️ Start Simulering", type="primary"):
            with GLOBAL_STATE_LOCK:
                global_state['simulation_active'] = True
                global_state['last_update_time'] = time.time() 
            st.rerun() 

    # --- TIMER OG PRISOPPDATERING (STABIL LOGIKK) ---
    WAIT_SECONDS = 60 
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
            st.rerun() 
        
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
    # Midlertidig kaller de uendrede funksjonene for å unngå NameError i eksemplet
    # Du må sørge for at disse er kopiert inn!
    def black_scholes_price(*args, **kwargs): return 0.0 
    def black_scholes_greeks(*args, **kwargs): return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
    def display_order_book(): st.empty()
    def display_market_info(): st.empty()
    def trading_interface(): st.empty()
    def display_portfolio(): st.empty()
    
    main()
