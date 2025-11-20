import numpy as np
from scipy.stats import norm
import streamlit as st
import time
import pandas as pd
import random
import threading 
import copy # Vi bruker copy.deepcopy for å unngå referanseproblemer

# ==============================================================================
# 1. INITIALISERING OG KJERNEFUNKSJONER
# ==============================================================================

class Student:
    """Enkel klasse for å holde Portefølje- og Kontantdata."""
    def __init__(self, student_id, role, initial_cash=100000.0):
        self.id = student_id
        self.role = role  
        self.cash = initial_cash
        self.portfolio = {'OP_C100': 0} # Holder antall opsjoner
        self.transactions = [] # Logg over gjennomførte handler

def initialize_state():
    """Initialiserer Streamlit session state."""
    if 'initialized' not in st.session_state:
        st.session_state.market_params = {
            'ticker': 'OP_C100',
            'S': 105.0,        
            'K': 100.0,       
            't': 0.25,        
            'r': 0.04,        
            'sigma': 0.30     
        }
        st.session_state.call_bids = []
        st.session_state.call_asks = []
        st.session_state.students = {}
        st.session_state.active_student = None
        st.session_state.user_role = None
        st.session_state.simulation_active = False
        st.session_state.last_update_time = time.time()
        st.session_state.status_message = None
        st.session_state.initialized = True
        st.session_state.order_counter = 0 
        st.session_state.trade_counter = 0 # Teller for unike transaksjons-IDer

# ... (update_stock_price, black_scholes_price, black_scholes_greeks forblir de samme)
def update_stock_price():
    market = st.session_state.market_params
    S, r, sigma = market['S'], market['r'], market['sigma']
    time_elapsed = time.time() - st.session_state.last_update_time
    delta_t_years = 1.0 / 525600.0 * time_elapsed 
    Z = np.random.standard_normal()
    dS = S * (r * delta_t_years + sigma * np.sqrt(delta_t_years) * Z)
    st.session_state.market_params['S'] = max(0.01, S + dS) 
    st.session_state.market_params['t'] = max(0, market['t'] - delta_t_years)

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
# ...

def submit_limit_order(student_id, option_type, side, price, quantity):
    """Legger inn en ordre direkte i listen i Session State."""
    # (Logikk uendret)
    if option_type != 'CALL':
        return False, "Kun CALL-opsjoner støttes."

    book_key = 'call_bids' if side == 'BID' else 'call_asks'
    
    st.session_state.order_counter += 1
    order_id = f"ORD_{st.session_state.order_counter}"
    
    order = {
        'order_id': order_id, 
        'id': student_id,
        'price': price,
        'quantity': quantity,
        'side': side,
        'time': time.time()
    }
    
    st.session_state[book_key].append(order)
    
    if side == 'BID':
        st.session_state[book_key].sort(key=lambda x: x['price'], reverse=True)
        return True, f"Limitordre lagt inn: BUD {quantity} @ {price:.2f} (ID: {order_id})"
    else: 
        st.session_state[book_key].sort(key=lambda x: x['price'])
        return True, f"Limitordre lagt inn: TILBUD {quantity} @ {price:.2f} (ID: {order_id})"


def cancel_order_by_id(order_id, student_id):
    """Fjerner en ordre fra ordreboken basert på ID."""
    # (Logikk uendret)
    keys = ['call_bids', 'call_asks']
    found = False
    
    for key in keys:
        new_list = []
        for order in st.session_state[key]:
            if order.get('order_id') == order_id and order.get('id') == student_id:
                found = True
            else:
                new_list.append(order)
        
        if found:
            st.session_state[key] = new_list
            if key == 'call_bids':
                st.session_state[key].sort(key=lambda x: x['price'], reverse=True)
            else:
                st.session_state[key].sort(key=lambda x: x['price'])
            return True
            
    return False

# ==============================================================================
# 2. MATCHINGSFUNKSJON (Kjernen i Market Order)
# ==============================================================================

def process_market_order(taker_id, side, quantity_remaining):
    """
    Gjennomfører en Market Order mot Limit Ordrer i boken.
    """
    taker = st.session_state.students[taker_id]
    
    if side == 'BUY':
        book_key = 'call_asks' # Kjøper matcher mot selgertilbud (ASKS)
        limit_book = st.session_state.call_asks
        order_type = 'ASK' # For transaksjonslogg
    else: # SELL
        book_key = 'call_bids' # Selger matcher mot kjøperbud (BIDS)
        limit_book = st.session_state.call_bids
        order_type = 'BID' # For transaksjonslogg

    # Lager en kopi for å iterere over mens vi muterer den originale boken
    temp_book = copy.deepcopy(limit_book) 
    new_limit_book = []
    
    filled_quantity = 0
    total_cost = 0

    for limit_order in temp_book:
        if quantity_remaining <= 0:
            new_limit_book.append(limit_order) # Legg tilbake resten av boken
            continue

        maker = st.session_state.students.get(limit_order['id'])
        if not maker:
            new_limit_book.append(limit_order)
            continue # Skip hvis maker ikke finnes

        qty_to_trade = min(quantity_remaining, limit_order['quantity'])
        trade_price = limit_order['price']
        trade_amount = qty_to_trade * trade_price

        # Sjekk taker sin kjøpekraft/posisjon FØR handelen
        if side == 'BUY' and taker.cash < trade_amount:
            # Hvis taker ikke har nok penger for neste ordre, stopp
            new_limit_book.append(limit_order) 
            break 
        if side == 'SELL' and taker.portfolio.get('OP_C100', 0) < quantity_remaining:
            return False, "Mangler nok opsjoner å selge!"

        # 1. Oppdatering av Market Taker (Trader)
        if side == 'BUY':
            taker.cash -= trade_amount
            taker.portfolio['OP_C100'] = taker.portfolio.get('OP_C100', 0) + qty_to_trade
        else: # SELL
            taker.cash += trade_amount
            taker.portfolio['OP_C100'] = taker.portfolio.get('OP_C100', 0) - qty_to_trade

        # 2. Oppdatering av Limit Maker (Market Maker)
        if side == 'BUY': # Limit Maker solgte til Trader
            maker.cash += trade_amount
            maker.portfolio['OP_C100'] = maker.portfolio.get('OP_C100', 0) - qty_to_trade
        else: # Limit Maker kjøpte fra Trader
            maker.cash -= trade_amount
            maker.portfolio['OP_C100'] = maker.portfolio.get('OP_C100', 0) + qty_to_trade

        # 3. Oppdater Market og Trade-logg
        quantity_remaining -= qty_to_trade
        filled_quantity += qty_to_trade
        total_cost += trade_amount
        
        st.session_state.trade_counter += 1
        trade_id = f"TRD_{st.session_state.trade_counter}"

        transaction_record = {
            'id': trade_id,
            'taker': taker_id,
            'maker': maker.id,
            'quantity': qty_to_trade,
            'price': trade_price,
            'time': time.time()
        }
        taker.transactions.append(transaction_record)
        maker.transactions.append(transaction_record) # Loggføres hos begge
        
        # 4. Oppdater resten av Limit Orderen
        if qty_to_trade < limit_order['quantity']:
            limit_order['quantity'] -= qty_to_trade
            new_limit_book.append(limit_order) # Delvis fylt ordre gjenstår
        # Hvis qty_to_trade == limit_order['quantity'], fylles ordren og den blir ikke lagt tilbake.

    # Sett den nye, oppdaterte boken tilbake i Session State
    st.session_state[book_key] = new_limit_book
    
    # Sortering etter oppdatering
    if side == 'BUY':
        st.session_state[book_key].sort(key=lambda x: x['price']) # ASKS (laveste først)
    else: 
        st.session_state[book_key].sort(key=lambda x: x['price'], reverse=True) # BIDS (høyeste først)

    if filled_quantity > 0:
        avg_price = total_cost / filled_quantity
        return True, f"Market Order utført: {filled_quantity} opsjoner @ {avg_price:.2f} i snitt. Gjenstår: {quantity_remaining}"
    elif quantity_remaining == quantity_remaining: # Fikk ikke fylt noe
        return False, f"Market Order feilet: Ordreboken er tom eller ingen tilgjengelige priser."
    else:
        return False, f"Market Order feilet: Ukjent feil."


# ... (cancel_order_by_id forblir den samme)

# ==============================================================================
# 3. UI-KOMPONENTER (Oppdatert Trading Interface)
# ==============================================================================

# ... (display_order_book, display_market_info forblir de samme)
def display_order_book():
    st.subheader("📚 Ordrebok Status (CALL)")
    bids = st.session_state.call_bids
    asks = st.session_state.call_asks
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
    market = st.session_state.market_params
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


def trading_interface():
    active_student = st.session_state.active_student
    
    if active_student.role == 'MAKER':
        st.subheader("✍️ Legg Inn Limit Order (Market Maker)")
        # ... (Limit Order logikk uendret) ...
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
                    st.session_state.status_message = {'type': 'success', 'content': msg}
                else:
                    st.session_state.status_message = {'type': 'error', 'content': msg}
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
                # Kaller Matchingsmotoren
                success, msg = process_market_order(active_student.id, side, quantity)
                
                if success:
                    st.session_state.status_message = {'type': 'success', 'content': msg}
                else:
                    st.session_state.status_message = {'type': 'error', 'content': msg}
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
    # ... (Kanselleringslogikk uendret) ...
    open_bids = [o for o in st.session_state.call_bids if o['id'] == student.id]
    open_asks = [o for o in st.session_state.call_asks if o['id'] == student.id]
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
                    st.session_state.status_message = {'type': 'success', 'content': f"Ordre {order_to_cancel} kansellert."}
                else:
                    st.session_state.status_message = {'type': 'error', 'content': f"Kansellering feilet for ID {order_to_cancel}. (Feil eier?)"}
                st.rerun() 
    else:
        st.info("Ingen åpne limitordrer.")


# ==============================================================================
# 4. HOVED APPLIKASJONSFLYT MED KONTROLLERT OPPDATERINGSLØKKE
# ==============================================================================

def main():
    st.set_page_config(page_title="Opsjonsmarked Simulator V8 (Market Order)", layout="wide")
    
    initialize_state()

    # --- LOGIN ---
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
                st.session_state.active_student = st.session_state.students[student_id]
                st.session_state.user_role = role_type
                st.rerun() 
        return

    st.title(f"🏛️ Opsjonsmarked Simulator (Rolle: {st.session_state.user_role})")
    
    # Statusmeldinger
    status_msg_placeholder = st.empty()
    if st.session_state.status_message:
        msg = st.session_state.status_message
        if msg['type'] == 'success':
            status_msg_placeholder.success(msg['content'])
        else:
            status_msg_placeholder.error(msg['content'])
        st.session_state.status_message = None

    # Sidefelt for kontroller
    sidebar_container = st.sidebar.container()
    sidebar_container.title("Kontroller")

    market_t = st.session_state.market_params['t']
    
    # Plassholder for timeren i sidepanelet
    timer_placeholder = sidebar_container.empty()
    
    # --- START/STOPP SIMULERING KNAPP ---
    if market_t > 0 and not st.session_state.simulation_active:
        if sidebar_container.button("▶️ Start Simulering", type="primary"):
            st.session_state.simulation_active = True
            st.rerun() 
    
    # --- UI RENDERING ---
    tab1, tab2, tab3 = st.tabs(["Markedsinnsikt", "Handelsplass", "Portefølje"])

    with tab1:
        display_market_info()

    with tab2:
        trading_interface()

    with tab3:
        display_portfolio()

    # --- KONTROLLERT OPPDATERINGSLØKKE ---
    WAIT_SECONDS = 60 # Pris oppdateres hvert 60. sekund

    if st.session_state.simulation_active and market_t > 0:
        
        sidebar_container.button("⏸️ Simulering pågår...", disabled=True)
        
        while st.session_state.simulation_active and market_t > 0:
            current_time = time.time()
            time_elapsed = current_time - st.session_state.last_update_time
            time_remaining = max(0, WAIT_SECONDS - time_elapsed)
            
            # 1. PRISOPPDATERING (hvis tiden er ute)
            if time_elapsed >= WAIT_SECONDS:
                
                update_stock_price()
                st.session_state.last_update_time = current_time
                st.session_state.status_message = {'type': 'success', 'content': f"Automatisk prisoppdatering fullført! Ny pris: ${st.session_state.market_params['S']:.2f}"}

                st.rerun() 
                
            # 2. NEDTELLING (oppdaterer kun sidepanelet)
            else:
                timer_placeholder.info(f"Pris oppdatering om {int(time_remaining) + 1} sekunder...")
                time.sleep(1) 
                
        # Hvis markedet har utløpt, stopper løkken
        sidebar_container.error("Opsjonen har utløpt.")

if __name__ == '__main__':
    main()