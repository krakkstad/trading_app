import numpy as np
from scipy.stats import norm
import streamlit as st
import time
import pandas as pd
import json 
from datetime import datetime
from pathlib import Path

# ==============================================================================
# 0. SHARED GLOBAL STATE
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
    """Get the truly global state shared across all sessions."""
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
            json.dump(data_to_save, f, indent=2)
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
    
    book = getattr(shared, book_key)
    book.append(order)
    
    # Sort order book (price-time priority)
    if side == 'BID':
        book.sort(key=lambda x: (-x['price'], x['time']))
    else:
        book.sort(key=lambda x: (x['price'], x['time']))
    
    return True, f"✅ Order placed: {side} {quantity} @ ${price:.2f}"

def cancel_order_by_id(order_id, student_id):
    """Cancel an order by ID."""
    shared = get_shared_state()
    
    for book_key in ['call_bids', 'call_asks']:
        book = getattr(shared, book_key)
        original_len = len(book)
        new_book = [o for o in book 
                   if not (o['order_id'] == order_id and o['id'] == student_id)]
        
        if len(new_book) < original_len:
            if book_key == 'call_bids':
                new_book.sort(key=lambda x: (-x['price'], x['time']))
            else:
                new_book.sort(key=lambda x: (x['price'], x['time']))
            
            setattr(shared, book_key, new_book)
            return True
    
    return False

def process_market_order(taker_id, side, quantity_requested):
    """
    Process a market order by matching with limit orders.
    NOW SUPPORTS SHORT SELLING - no position check!
    """
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
        
        # Check if taker has sufficient CASH (only restriction)
        if side == 'BUY':
            if taker.cash < trade_amount:
                new_book.append(limit_order)
                break
        # REMOVED: No check for selling - short selling is allowed!
        
        # Execute trade
        if side == 'BUY':
            taker.cash -= trade_amount
            taker.portfolio['OP_C100'] = taker.portfolio.get('OP_C100', 0) + qty_to_trade
            maker.cash += trade_amount
            maker.portfolio['OP_C100'] = maker.portfolio.get('OP_C100', 0) - qty_to_trade
        else:  # SELL (can go negative = short position)
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
    
    # Update order book
    setattr(shared, book_key, new_book)
    
    # Save to file periodically
    current_time = time.time()
    if current_time - shared.last_file_save > 10:  # Save every 10 seconds (reduced frequency)
        save_students_to_file(students)
        shared.last_file_save = current_time
    
    if filled_quantity > 0:
        avg_price = total_cost / filled_quantity
        msg = f"✅ Filled {filled_quantity} @ ${avg_price:.2f}"
        if quantity_remaining > 0:
            msg += f" ({quantity_remaining} unfilled)"
        return True, msg
    else:
        return False, "❌ No liquidity available"

# ==============================================================================
# 3. UI COMPONENTS
# ==============================================================================

def initialize_session_state():
    """Initialize session-specific state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.active_student_id = None
        st.session_state.user_role = None
        st.session_state.status_message = None
        st.session_state.initialized = True

def get_active_student():
    """Get the currently active student object."""
    if st.session_state.active_student_id:
        shared = get_shared_state()
        return shared.students.get(st.session_state.active_student_id)
    return None

def display_order_book():
    """Display the current order book."""
    shared = get_shared_state()
    bids = shared.call_bids
    asks = shared.call_asks
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🟢 BIDS**")
        if bids:
            df_bids = pd.DataFrame(bids[:5])
            df_bids = df_bids[['price', 'quantity', 'id']]
            df_bids.columns = ['Price', 'Qty', 'Maker']
            st.dataframe(df_bids, use_container_width=True, hide_index=True)
        else:
            st.info("No bids")
    
    with col2:
        st.markdown("**🔴 ASKS**")
        if asks:
            df_asks = pd.DataFrame(asks[:5])
            df_asks = df_asks[['price', 'quantity', 'id']]
            df_asks.columns = ['Price', 'Qty', 'Maker']
            st.dataframe(df_asks, use_container_width=True, hide_index=True)
        else:
            st.info("No asks")
    
    # Show spread
    if bids and asks:
        best_bid = bids[0]['price']
        best_ask = asks[0]['price']
        spread = best_ask - best_bid
        
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Best Bid", f"${best_bid:.2f}")
        col_s2.metric("Best Ask", f"${best_ask:.2f}")
        col_s3.metric("Spread", f"${spread:.2f}")

def display_market_info():
    """Display current market parameters and pricing."""
    shared = get_shared_state()
    market = shared.market_params
    S, K, t, r, sigma = market['S'], market['K'], market['t'], market['r'], market['sigma']
    
    # Compact market params
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Stock (S)", f"${S:.2f}")
    col2.metric("Strike (K)", f"${K:.2f}")
    col3.metric("Time (t)", f"{t:.4f}y")
    col4.metric("Rate (r)", f"{r*100:.1f}%")
    col5.metric("Vol (σ)", f"{sigma*100:.0f}%")
    
    # Fair values
    call_price = black_scholes_price(S, K, t, r, sigma, 'call')
    call_greeks = black_scholes_greeks(S, K, t, r, sigma, 'call')
    
    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
    col_p1.metric("Fair Value", f"${call_price:.3f}")
    col_p2.metric("Δ", f"{call_greeks['delta']:.3f}")
    col_p3.metric("Γ", f"{call_greeks['gamma']:.5f}")
    col_p4.metric("Θ", f"{call_greeks['theta']:.4f}")
    col_p5.metric("ν", f"{call_greeks['vega']:.3f}")
    
    st.markdown("---")
    display_order_book()

def trading_interface():
    """Display trading interface based on user role."""
    student = get_active_student()
    if not student:
        st.error("Student not found!")
        return
    
    # Show portfolio summary at top
    position = student.portfolio.get('OP_C100', 0)
    position_color = "🔴" if position < 0 else "🟢" if position > 0 else "⚪"
    
    col_top1, col_top2 = st.columns(2)
    col_top1.metric("💰 Cash", f"${student.cash:.2f}")
    col_top2.metric(f"{position_color} Position", f"{position:+d} contracts", 
                    help="Negative = Short position")
    
    if student.role == 'MAKER':
        st.subheader("Market Maker: Submit Limit Order")
        
        with st.form("maker_form", clear_on_submit=True):
            col1, col2, col3 = st.columns(3)
            
            side = col1.selectbox("Side", ['BID', 'ASK'], key='m_side')
            price = col2.number_input("Price", min_value=0.01, value=10.0, 
                                     format="%.2f", key="m_price")
            quantity = col3.number_input("Qty", min_value=1, value=10, 
                                        step=1, key="m_qty")
            
            submitted = st.form_submit_button("📝 Submit Order", type="primary", use_container_width=True)
            
            if submitted:
                success, msg = submit_limit_order(student.id, 'CALL', side, price, quantity)
                st.session_state.status_message = {
                    'type': 'success' if success else 'error',
                    'content': msg
                }
                st.rerun()
    
    elif student.role == 'TRADER':
        st.subheader("Trader: Submit Market Order")
        st.caption("ℹ️ Short selling enabled - you can sell even with 0 or negative position")
        
        with st.form("trader_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            side = col1.selectbox("Side", ['BUY', 'SELL'], key='t_side')
            quantity = col2.number_input("Qty", min_value=1, value=5, 
                                        step=1, key="t_qty")
            
            submitted = st.form_submit_button(f"🚀 {side}", type="primary", use_container_width=True)
            
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
    
    st.subheader(f"Portfolio: {student.id}")
    
    # Portfolio summary
    position = student.portfolio.get('OP_C100', 0)
    
    # Calculate mark-to-market
    shared = get_shared_state()
    market = shared.market_params
    current_option_value = black_scholes_price(
        market['S'], market['K'], market['t'], market['r'], market['sigma'], 'call'
    )
    
    position_value = position * current_option_value
    total_value = student.cash + position_value
    initial_value = 100000.0 if student.role == 'MAKER' else 50000.0
    pnl = total_value - initial_value
    pnl_pct = (pnl / initial_value) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Cash", f"${student.cash:.2f}")
    
    # Show position with color coding
    if position < 0:
        col2.metric("📊 Position", f"{position} (SHORT)", delta="⚠️ Short", delta_color="inverse")
    elif position > 0:
        col2.metric("📊 Position", f"{position} (LONG)", delta="✓ Long", delta_color="normal")
    else:
        col2.metric("📊 Position", f"{position} (FLAT)")
    
    col3.metric("📈 Total Value", f"${total_value:.2f}")
    col4.metric("💹 P&L", f"${pnl:.2f}", f"{pnl_pct:+.2f}%")
    
    if position != 0:
        st.info(f"Position Value: {position} × ${current_option_value:.3f} = ${position_value:.2f}")
    
    st.markdown("---")
    
    # Transaction history (last 10)
    st.subheader("Recent Transactions")
    if student.transactions:
        df_trades = pd.DataFrame(student.transactions[-10:][::-1])  # Last 10, reversed
        cols = ['time', 'side', 'quantity', 'price', 'maker', 'taker']
        df_trades = df_trades[[c for c in cols if c in df_trades.columns]]
        st.dataframe(df_trades, use_container_width=True, hide_index=True)
    else:
        st.info("No transactions yet")
    
    # Open orders
    st.markdown("---")
    st.subheader("Open Orders")
    shared = get_shared_state()
    open_bids = [o for o in shared.call_bids if o['id'] == student.id]
    open_asks = [o for o in shared.call_asks if o['id'] == student.id]
    open_orders = open_bids + open_asks
    
    if open_orders:
        df_orders = pd.DataFrame(open_orders)
        df_orders = df_orders[['order_id', 'side', 'price', 'quantity']]
        st.dataframe(df_orders, use_container_width=True, hide_index=True)
        
        with st.form("cancel_form"):
            order_to_cancel = st.selectbox(
                "Cancel Order", 
                [o['order_id'] for o in open_orders]
            )
            cancel_btn = st.form_submit_button("❌ Cancel", type="secondary")
            
            if cancel_btn:
                success = cancel_order_by_id(order_to_cancel, student.id)
                msg = f"Cancelled {order_to_cancel}" if success else "Cancel failed"
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
        layout="wide"
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
        
        with st.form("login_form"):
            col1, col2 = st.columns(2)
            student_id = col1.text_input("Student ID", value="")
            role = col2.selectbox("Role", ['MAKER', 'TRADER'])
            
            login_btn = st.form_submit_button("🚀 Start Trading", type="primary", use_container_width=True)
            
            if login_btn and student_id:
                if student_id not in shared.students:
                    # Create new student
                    initial_cash = 100000.0 if role == 'MAKER' else 50000.0
                    shared.students[student_id] = Student(student_id, role, cash=initial_cash)
                    save_students_to_file(shared.students)
                
                # Set active student
                st.session_state.active_student_id = student_id
                st.session_state.user_role = shared.students[student_id].role
                st.query_params["user_id"] = student_id
                st.rerun()
        
        return
    
    # ========== MAIN INTERFACE ==========
    
    # Header
    col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
    col_h1.title(f"📈 Options Market")
    col_h2.write("")
    col_h2.caption(f"**{st.session_state.active_student_id}** ({st.session_state.user_role})")
    
    # Display status messages
    if st.session_state.status_message:
        msg = st.session_state.status_message
        msg_container = st.empty()
        if msg['type'] == 'success':
            msg_container.success(msg['content'])
        else:
            msg_container.error(msg['content'])
        st.session_state.status_message = None
    
    # ========== SIDEBAR CONTROLS ==========
    with st.sidebar:
        st.title("⚙️ Controls")
        
        # Simulation control
        market_t = shared.market_params['t']
        sim_active = shared.simulation_active
        
        if market_t <= 0:
            st.error("⚠️ Expired")
        elif not sim_active:
            if st.button("▶️ Start Sim", use_container_width=True):
                shared.simulation_active = True
                shared.last_update_time = time.time()
                st.rerun()
        else:
            if st.button("⏸️ Pause", use_container_width=True):
                shared.simulation_active = False
                st.rerun()
            
            # Timer
            WAIT_SECONDS = 60
            time_elapsed = time.time() - shared.last_update_time
            
            if time_elapsed >= WAIT_SECONDS:
                update_stock_price()
                st.rerun()
            else:
                time_remaining = WAIT_SECONDS - time_elapsed
                st.caption(f"⏳ Update in {int(time_remaining)}s")
        
        st.markdown("---")
        
        # Quick actions
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.active_student_id = None
            st.session_state.user_role = None
            st.query_params.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Stats
        st.caption(f"📊 Orders: {len(shared.call_bids)} bids, {len(shared.call_asks)} asks")
        st.caption(f"👥 Students: {len(shared.students)}")
        st.caption(f"💱 Trades: {shared.trade_counter}")
    
    # ========== MAIN TABS ==========
    tab1, tab2, tab3 = st.tabs(["📊 Market", "💱 Trade", "📁 Portfolio"])
    
    with tab1:
        display_market_info()
    
    with tab2:
        trading_interface()
    
    with tab3:
        display_portfolio()

if __name__ == '__main__':
    main()
