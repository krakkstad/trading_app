import numpy as np
from scipy.stats import norm
import streamlit as st
import time
import pandas as pd
import copy 
import json 
from datetime import datetime
from pathlib import Path

# ==============================================================================
# 0. SHARED GLOBAL STATE (CRITICAL FIX!)
# ==============================================================================

class GlobalState:
    """Global state container that can be safely cached and modified."""
    def __init__(self):
        self.market_params = {
            'ticker': 'OP_C100', 
            'S': 105.0, 
            'K': 100.0, 
            't': 0.25, 
            'r': 0.04, 
            'sigma': 0.30     
        }
        self.call_bids = []
        self.call_asks = []
        self.last_update_time = time.time()
        self.simulation_active = False
        self.order_counter = 0
        self.trade_counter = 0
        self.students = {}
        self.last_file_save = time.time()

@st.cache_resource
def get_shared_state():
    """
    CRITICAL: Using @st.cache_resource creates a TRULY GLOBAL state
    that persists across ALL user sessions and browser tabs.
    This is the key to multi-user synchronization!
    """
    return GlobalState()

STUDENTS_FILE = "students.json"

# ==============================================================================
# 1. STUDENT CLASS AND FILE OPERATIONS
# ==============================================================================

class Student:
    """Student portfolio and cash holdings."""
    def __init__(self, student_id, role, cash=100000.0, portfolio=None, transactions=None):
        self.id = student_id
        self.role = role  
        self.cash = cash
        self.portfolio = portfolio if portfolio is not None else {'OP_C100': 0} 
        self.transactions = transactions if transactions is not None else [] 
    
    def to_dict(self):
        return {
            'id': self.id,
            'role': self.role,
            'cash': self.cash,
            'portfolio': self.portfolio,
            'transactions': self.transactions
        }
    
    @staticmethod
    def from_dict(data):
        return Student(
            student_id=data['id'], 
            role=data['role'], 
            cash=data['cash'], 
            portfolio=data.get('portfolio', {'OP_C100': 0}), 
            transactions=data.get('transactions', [])
        )

def load_students_from_file():
    """Load students from persistent storage."""
    try:
        if Path(STUDENTS_FILE).exists():
            with open(STUDENTS_FILE, 'r') as f:
                data = json.load(f)
                return {k: Student.from_dict(v) for k, v in data.items()}
    except Exception as e:
        st.error(f"Error loading students: {e}")
    return {}

def save_students_to_file(students_dict):
    """Save students to persistent storage."""
    try:
        data_to_save = {k: v.to_dict() for k, v in students_dict.items()}
        with open(STUDENTS_FILE, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        return True
    except Exception as e:
        st.error(f"Error saving students: {e}")
        return False

def initialize_students():
    """Load students into shared state if not already loaded."""
    shared = get_shared_state()
    if not shared.students:
        shared.students = load_students_from_file()

# ==============================================================================
# 2. CORE TRADING FUNCTIONS
# ==============================================================================

def update_stock_price():
    """Update stock price using geometric Brownian motion."""
    shared = get_shared_state()
    market = shared.market_params
    S, r, sigma = market['S'], market['r'], market['sigma']
    
    time_elapsed = time.time() - shared.last_update_time
    delta_t_years = 60.0 / 525600.0  # 60 seconds in years
    
    Z = np.random.standard_normal()
    dS = S * (r * delta_t_years + sigma * np.sqrt(delta_t_years) * Z)
    
    shared.market_params['S'] = max(0.01, S + dS)
    shared.market_params['t'] = max(0, market['t'] - delta_t_years)
    shared.last_update_time = time.time()

def black_scholes_price(S, K, t, r, sigma, option_type='call'):
    """Calculate Black-Scholes option price."""
    if t <= 0: 
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def black_scholes_greeks(S, K, t, r, sigma, option_type='call'):
    """Calculate option Greeks."""
    if t <= 0: 
        return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    N_prime_d1 = norm.pdf(d1)
    
    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    gamma = N_prime_d1 / (S * sigma * np.sqrt(t))
    vega = S * N_prime_d1 * np.sqrt(t) * 0.01
    
    # Simplified theta calculation
    if option_type == 'call':
        theta = (-S * N_prime_d1 * sigma / (2 * np.sqrt(t)) 
                 - r * K * np.exp(-r * t) * norm.cdf(d2)) / 365
    else:
        theta = (-S * N_prime_d1 * sigma / (2 * np.sqrt(t)) 
                 + r * K * np.exp(-r * t) * norm.cdf(-d2)) / 365
    
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}

def submit_limit_order(student_id, option_type, side, price, quantity):
    """Submit a limit order to the order book."""
    if option_type != 'CALL': 
        return False, "Only CALL options supported."
    
    shared = get_shared_state()
    book_key = 'call_bids' if side == 'BID' else 'call_asks'
    
    shared.order_counter += 1
    order_id = f"ORD_{shared.order_counter}"
    
    order = {
        'order_id': order_id, 
        'id': student_id, 
        'price': float(price), 
        'quantity': int(quantity), 
        'side': side, 
        'time': time.time()
    }
    
    # Use getattr to access the attribute dynamically
    book = getattr(shared, book_key)
    book.append(order)
    
    # Sort order book
    if side == 'BID':
        book.sort(key=lambda x: (-x['price'], x['time']))
    else:
        book.sort(key=lambda x: (x['price'], x['time']))
    
    return True, f"Limit order placed: {side} {quantity} @ {price:.2f} (ID: {order_id})"

def cancel_order_by_id(order_id, student_id):
    """Cancel an order by ID."""
    shared = get_shared_state()
    
    for book_key in ['call_bids', 'call_asks']:
        book = getattr(shared, book_key)
        original_len = len(book)
        new_book = [o for o in book 
                   if not (o['order_id'] == order_id and o['id'] == student_id)]
        
        if len(new_book) < original_len:
            # Re-sort after removal
            if book_key == 'call_bids':
                new_book.sort(key=lambda x: (-x['price'], x['time']))
            else:
                new_book.sort(key=lambda x: (x['price'], x['time']))
            
            setattr(shared, book_key, new_book)
            return True
    
    return False

def process_market_order(taker_id, side, quantity_requested):
    """Process a market order by matching with limit orders."""
    shared = get_shared_state()
    students = shared.students
    
    taker = students.get(taker_id)
    if not taker:
        return False, "Taker not found."
    
    # Determine which book to trade against
    book_key = 'call_asks' if side == 'BUY' else 'call_bids'
    
    quantity_remaining = quantity_requested
    filled_quantity = 0
    total_cost = 0
    new_book = []
    
    # Get the order book using getattr
    current_book = getattr(shared, book_key)
    
    for limit_order in current_book:
        if quantity_remaining <= 0:
            new_book.append(limit_order)
            continue
        
        maker_id = limit_order['id']
        maker = students.get(maker_id)
        
        if not maker:
            new_book.append(limit_order)
            continue
        
        # Determine trade quantity
        qty_to_trade = min(quantity_remaining, limit_order['quantity'])
        trade_price = limit_order['price']
        trade_amount = qty_to_trade * trade_price
        
        # Check if taker has sufficient funds/options
        if side == 'BUY':
            if taker.cash < trade_amount:
                new_book.append(limit_order)
                break
        else:  # SELL
            if taker.portfolio.get('OP_C100', 0) < qty_to_trade:
                return False, f"Insufficient options to sell! You have {taker.portfolio.get('OP_C100', 0)}, need {quantity_requested}"
        
        # Execute trade
        if side == 'BUY':
            taker.cash -= trade_amount
            taker.portfolio['OP_C100'] = taker.portfolio.get('OP_C100', 0) + qty_to_trade
            maker.cash += trade_amount
            maker.portfolio['OP_C100'] = maker.portfolio.get('OP_C100', 0) - qty_to_trade
        else:  # SELL
            taker.cash += trade_amount
            taker.portfolio['OP_C100'] = taker.portfolio.get('OP_C100', 0) - qty_to_trade
            maker.cash -= trade_amount
            maker.portfolio['OP_C100'] = maker.portfolio.get('OP_C100', 0) + qty_to_trade
        
        quantity_remaining -= qty_to_trade
        filled_quantity += qty_to_trade
        total_cost += trade_amount
        
        # Record transaction
        shared.trade_counter += 1
        trade_id = f"TRD_{shared.trade_counter}"
        
        transaction_record = {
            'trade_id': trade_id,
            'taker': taker_id,
            'maker': maker_id,
            'quantity': qty_to_trade,
            'price': trade_price,
            'time': datetime.now().strftime('%H:%M:%S'),
            'side': side
        }
        
        taker.transactions.append(transaction_record)
        maker.transactions.append(transaction_record)
        
        # Update limit order or remove if filled
        if qty_to_trade < limit_order['quantity']:
            limit_order['quantity'] -= qty_to_trade
            new_book.append(limit_order)
    
    # Update order book using setattr
    setattr(shared, book_key, new_book)
    
    # Save to file periodically (not on every trade to avoid I/O overhead)
    current_time = time.time()
    if current_time - shared.last_file_save > 5:  # Save every 5 seconds max
        save_students_to_file(students)
        shared.last_file_save = current_time
    
    if filled_quantity > 0:
        avg_price = total_cost / filled_quantity
        msg = f"Market order executed: {filled_quantity} contracts @ ${avg_price:.2f} avg."
        if quantity_remaining > 0:
            msg += f" ({quantity_remaining} unfilled)"
        return True, msg
    else:
        return False, "Market order failed: No liquidity available at current prices."

# ==============================================================================
# 3. UI COMPONENTS
# ==============================================================================

def initialize_session_state():
    """Initialize session-specific state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.active_student_id = None
        st.session_state.user_role = None
        st.session_state.status_message = None
        st.session_state.auto_refresh = True
        st.session_state.initialized = True

def get_active_student():
    """Get the currently active student object."""
    if st.session_state.active_student_id:
        shared = get_shared_state()
        return shared.students.get(st.session_state.active_student_id)
    return None

def display_order_book():
    """Display the current order book."""
    st.subheader("📚 Order Book (CALL Options)")
    
    shared = get_shared_state()
    bids = shared.call_bids
    asks = shared.call_asks
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🟢 BIDS (Buy Orders)**")
        if bids:
            df_bids = pd.DataFrame(bids[:10])  # Show top 10
            df_bids = df_bids[['price', 'quantity', 'id', 'order_id']]
            df_bids.columns = ['Price', 'Quantity', 'Maker', 'Order ID']
            st.dataframe(df_bids, use_container_width=True, hide_index=True)
        else:
            st.info("No buy orders")
    
    with col2:
        st.markdown("**🔴 ASKS (Sell Orders)**")
        if asks:
            df_asks = pd.DataFrame(asks[:10])  # Show top 10
            df_asks = df_asks[['price', 'quantity', 'id', 'order_id']]
            df_asks.columns = ['Price', 'Quantity', 'Maker', 'Order ID']
            st.dataframe(df_asks, use_container_width=True, hide_index=True)
        else:
            st.info("No sell orders")
    
    # Show best bid/ask
    best_bid = bids[0]['price'] if bids else None
    best_ask = asks[0]['price'] if asks else None
    
    col_spread = st.columns(3)
    if best_bid:
        col_spread[0].metric("Best Bid", f"${best_bid:.2f}")
    if best_ask:
        col_spread[1].metric("Best Ask", f"${best_ask:.2f}")
    if best_bid and best_ask:
        spread = best_ask - best_bid
        col_spread[2].metric("Spread", f"${spread:.2f}")

def display_market_info():
    """Display current market parameters and pricing."""
    shared = get_shared_state()
    market = shared.market_params
    S, K, t, r, sigma = market['S'], market['K'], market['t'], market['r'], market['sigma']
    
    st.subheader("⚙️ Market Parameters")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Underlying Price (S)", f"${S:.2f}")
    col2.metric("Strike Price (K)", f"${K:.2f}")
    col3.metric("Time to Expiry", f"{t:.4f} years")
    
    col4, col5 = st.columns(2)
    col4.metric("Risk-free Rate", f"{r*100:.2f}%")
    col5.metric("Volatility (σ)", f"{sigma*100:.2f}%")
    
    # Calculate fair values
    call_price = black_scholes_price(S, K, t, r, sigma, 'call')
    put_price = black_scholes_price(S, K, t, r, sigma, 'put')
    call_greeks = black_scholes_greeks(S, K, t, r, sigma, 'call')
    
    st.subheader("💰 Black-Scholes Fair Values")
    col_price1, col_price2 = st.columns(2)
    col_price1.metric("Call Option", f"${call_price:.4f}")
    col_price2.metric("Put Option", f"${put_price:.4f}")
    
    st.subheader("📐 Option Greeks (Call)")
    col_g1, col_g2, col_g3, col_g4 = st.columns(4)
    col_g1.metric("Delta (Δ)", f"{call_greeks['delta']:.4f}")
    col_g2.metric("Gamma (Γ)", f"{call_greeks['gamma']:.6f}")
    col_g3.metric("Theta (Θ)", f"{call_greeks['theta']:.4f}")
    col_g4.metric("Vega (ν)", f"{call_greeks['vega']:.4f}")
    
    st.markdown("---")
    display_order_book()

def trading_interface():
    """Display trading interface based on user role."""
    student = get_active_student()
    if not student:
        st.error("Student not found!")
        return
    
    if student.role == 'MAKER':
        st.subheader("✍️ Submit Limit Order (Market Maker)")
        st.info(f"💰 Cash: ${student.cash:.2f} | 📊 Options: {student.portfolio.get('OP_C100', 0)}")
        
        with st.form("maker_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            option_type = col1.selectbox("Option Type", ['CALL'], key='maker_opt')
            side = col2.selectbox("Order Side", ['BID', 'ASK'], key='maker_side')
            
            col3, col4 = st.columns(2)
            price = col3.number_input("Limit Price", min_value=0.01, value=10.0, 
                                     format="%.2f", key="maker_price")
            quantity = col4.number_input("Quantity", min_value=1, value=1, 
                                        step=1, key="maker_qty")
            
            submitted = st.form_submit_button("📝 Submit Limit Order", type="primary")
            
            if submitted:
                success, msg = submit_limit_order(student.id, option_type, side, price, quantity)
                st.session_state.status_message = {
                    'type': 'success' if success else 'error',
                    'content': msg
                }
                st.rerun()
    
    elif student.role == 'TRADER':
        st.subheader("💸 Submit Market Order (Trader)")
        st.info(f"💰 Cash: ${student.cash:.2f} | 📊 Options: {student.portfolio.get('OP_C100', 0)}")
        
        with st.form("trader_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            side = col1.selectbox("Order Side", ['BUY', 'SELL'], key='trader_side')
            quantity = col2.number_input("Quantity", min_value=1, value=1, 
                                        step=1, key="trader_qty")
            
            submitted = st.form_submit_button(f"🚀 Execute Market {side}", type="primary")
            
            if submitted:
                success, msg = process_market_order(student.id, side, quantity)
                st.session_state.status_message = {
                    'type': 'success' if success else 'error',
                    'content': msg
                }
                st.rerun()

def display_portfolio():
    """Display student's portfolio and transaction history."""
    student = get_active_student()
    if not student:
        st.error("Student not found!")
        return
    
    st.subheader(f"👤 Portfolio: {student.id} ({student.role})")
    
    col1, col2 = st.columns(2)
    col1.metric("💰 Cash Balance", f"${student.cash:.2f}")
    col2.metric("📊 Options Position", f"{student.portfolio.get('OP_C100', 0)} contracts")
    
    # Calculate P&L
    shared = get_shared_state()
    market = shared.market_params
    current_option_value = black_scholes_price(
        market['S'], market['K'], market['t'], market['r'], market['sigma'], 'call'
    )
    
    portfolio_value = student.cash + (student.portfolio.get('OP_C100', 0) * current_option_value)
    initial_value = 100000.0 if student.role == 'MAKER' else 50000.0
    pnl = portfolio_value - initial_value
    pnl_pct = (pnl / initial_value) * 100
    
    col3, col4 = st.columns(2)
    col3.metric("📈 Portfolio Value", f"${portfolio_value:.2f}")
    col4.metric("💹 P&L", f"${pnl:.2f}", f"{pnl_pct:+.2f}%")
    
    st.markdown("---")
    
    # Transaction history
    st.subheader("📋 Transaction History")
    if student.transactions:
        df_trades = pd.DataFrame(student.transactions)
        # Reorder columns for better display
        cols = ['time', 'trade_id', 'side', 'quantity', 'price', 'taker', 'maker']
        df_trades = df_trades[[c for c in cols if c in df_trades.columns]]
        df_trades.columns = ['Time', 'Trade ID', 'Side', 'Qty', 'Price', 'Taker', 'Maker']
        st.dataframe(df_trades, use_container_width=True, hide_index=True)
    else:
        st.info("No transactions yet")
    
    st.markdown("---")
    
    # Open orders
    st.subheader("🛑 Open Orders")
    shared = get_shared_state()
    open_bids = [o for o in shared.call_bids if o['id'] == student.id]
    open_asks = [o for o in shared.call_asks if o['id'] == student.id]
    open_orders = open_bids + open_asks
    
    if open_orders:
        df_orders = pd.DataFrame(open_orders)
        df_orders = df_orders[['order_id', 'side', 'price', 'quantity']]
        df_orders.columns = ['Order ID', 'Side', 'Price', 'Quantity']
        st.dataframe(df_orders, use_container_width=True, hide_index=True)
        
        with st.form("cancel_form"):
            order_to_cancel = st.selectbox(
                "Select Order to Cancel", 
                [o['order_id'] for o in open_orders]
            )
            cancel_btn = st.form_submit_button("❌ Cancel Order", type="secondary")
            
            if cancel_btn:
                success = cancel_order_by_id(order_to_cancel, student.id)
                msg = f"Order {order_to_cancel} cancelled" if success else "Cancellation failed"
                st.session_state.status_message = {
                    'type': 'success' if success else 'error',
                    'content': msg
                }
                st.rerun()
    else:
        st.info("No open orders")

# ==============================================================================
# 4. MAIN APPLICATION
# ==============================================================================

def main():
    st.set_page_config(
        page_title="Options Market Simulator",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize
    initialize_session_state()
    initialize_students()
    
    shared = get_shared_state()
    
    # ========== LOGIN LOGIC ==========
    if st.session_state.active_student_id is None:
        # Check URL parameters
        query_params = st.query_params
        user_id_from_url = query_params.get("user_id")
        
        if user_id_from_url and user_id_from_url in shared.students:
            st.session_state.active_student_id = user_id_from_url
            st.session_state.user_role = shared.students[user_id_from_url].role
            st.rerun()
        
        # Show login form
        st.title("🏛️ Options Market Simulator")
        st.markdown("### Login or Register")
        
        with st.form("login_form"):
            student_id = st.text_input("Student ID", value="Student_A")
            role = st.selectbox("Role", ['MAKER', 'TRADER'])
            
            col1, col2 = st.columns(2)
            login_btn = col1.form_submit_button("🔐 Login", type="primary")
            register_btn = col2.form_submit_button("📝 Register", type="secondary")
            
            if login_btn or register_btn:
                if student_id not in shared.students:
                    # Create new student
                    initial_cash = 100000.0 if role == 'MAKER' else 50000.0
                    shared.students[student_id] = Student(student_id, role, cash=initial_cash)
                    save_students_to_file(shared.students)
                    st.success(f"✅ Registered new {role}: {student_id}")
                
                # Set active student
                st.session_state.active_student_id = student_id
                st.session_state.user_role = shared.students[student_id].role
                st.query_params["user_id"] = student_id
                st.rerun()
        
        # Show existing students
        if shared.students:
            st.markdown("---")
            st.markdown("### 👥 Existing Students")
            students_df = pd.DataFrame([
                {'ID': k, 'Role': v.role, 'Cash': f"${v.cash:.2f}", 
                 'Options': v.portfolio.get('OP_C100', 0)}
                for k, v in shared.students.items()
            ])
            st.dataframe(students_df, use_container_width=True, hide_index=True)
        
        return
    
    # ========== MAIN INTERFACE ==========
    
    # Title and status
    st.title(f"🏛️ Options Market Simulator")
    st.caption(f"Logged in as: **{st.session_state.active_student_id}** ({st.session_state.user_role})")
    
    # Display status messages
    if st.session_state.status_message:
        msg = st.session_state.status_message
        if msg['type'] == 'success':
            st.success(msg['content'])
        else:
            st.error(msg['content'])
        st.session_state.status_message = None
    
    # ========== SIDEBAR CONTROLS ==========
    with st.sidebar:
        st.title("🎛️ Controls")
        
        # Simulation control
        market_t = shared.market_params['t']
        sim_active = shared.simulation_active
        
        st.subheader("⏱️ Market Simulation")
        
        if market_t <= 0:
            st.error("⚠️ Option has expired!")
        elif not sim_active:
            if st.button("▶️ Start Simulation", type="primary", use_container_width=True):
                shared.simulation_active = True
                shared.last_update_time = time.time()
                st.rerun()
        else:
            if st.button("⏸️ Pause Simulation", type="secondary", use_container_width=True):
                shared.simulation_active = False
                st.rerun()
            
            # Timer display
            WAIT_SECONDS = 60
            time_elapsed = time.time() - shared.last_update_time
            
            if time_elapsed >= WAIT_SECONDS:
                update_stock_price()
                st.success(f"✅ Price updated: ${shared.market_params['S']:.2f}")
                st.rerun()
            else:
                time_remaining = WAIT_SECONDS - time_elapsed
                st.info(f"⏳ Next update in: {int(time_remaining)+1}s")
        
        st.markdown("---")
        
        # Auto-refresh toggle
        st.subheader("🔄 Auto-Refresh")
        auto_refresh = st.checkbox(
            "Enable auto-refresh (5s)", 
            value=st.session_state.auto_refresh,
            help="Automatically refresh to see other users' orders"
        )
        st.session_state.auto_refresh = auto_refresh
        
        if auto_refresh:
            st.caption("Auto-refreshing every 5 seconds...")
            time.sleep(5)
            st.rerun()
        
        # Manual refresh button
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()
        
        st.markdown("---")
        
        # Logout
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.active_student_id = None
            st.session_state.user_role = None
            st.query_params.clear()
            st.rerun()
        
        # Admin controls
        st.markdown("---")
        st.subheader("⚙️ Admin")
        if st.button("💾 Force Save", help="Save all data to file"):
            save_students_to_file(shared.students)
            st.success("Data saved!")
    
    # ========== MAIN TABS ==========
    tab1, tab2, tab3 = st.tabs(["📊 Market Info", "💱 Trading", "📁 Portfolio"])
    
    with tab1:
        display_market_info()
    
    with tab2:
        trading_interface()
    
    with tab3:
        display_portfolio()

if __name__ == '__main__':
    main()
