
# 2x3 Network Environment Analysis `envs/net_2x3.py`

This document provides a detailed breakdown of the `envs/net_2x3.py` environment for your thesis methodology section.

## 1. The "Dashboard" (State Space)

**Function:** `get_step_obs(self, action)` (Lines 232-262)

**What the AI sees:**
The state space is **Partially Observed**. Each agent sees only its own local information plus some information about its direct upstream and downstream connections. It does *not* see the full global state of all other agents.

Specifically, each agent observes a vector containing:
1.  **`self.inventory`**: Its current on-hand inventory level.
2.  **`self.backlog`**: The total backlog (unmet orders) it owes to its downstream customers.
    *   *Note:* The code sums the backlog for the two streams it manages: `self.backlog[i*2+j][0] + self.backlog[i*2+j][1]`.
3.  **Upstream Backlog**: The backlog that its *own suppliers* (at the level above) have accumulated.
    *   This tells the agent if its suppliers are struggling to fulfill orders.
4.  **Downstream Actions (Demand)**: The total orders placed by the agents directly below it in the previous step.
    *   This represents the "incoming demand" signal.
5.  **`self.order`**: The pipeline of orders that are currently "on the truck."
    *   This is a list of orders placed in the past `LEAD_TIME` steps that haven't arrived yet.

**Thesis Description:**
"The environment employs a partially observable Markov decision process (POMDP). Each agent's observation space consists of its local inventory position, its current backlog of unfilled orders, the backlog status of its upstream suppliers (providing visibility into supply reliability), the most recent demand signal from downstream entities, and a vector of incoming shipments currently in transit (the supply line)."

## 2. The "Controls" (Action Space)

**Function:** `step()` and `action_map()` (Lines 192-209)

**What the input represents:**
The agent outputs a single **Discrete Integer** between 0 and 24.

**Interpretation:**
This single number controls **two independent order quantities** simultaneously—one for each of the two suppliers at the level above.
*   **Formula:**
    *   `Order to Supplier A = Action // 5` (Integer Division)
    *   `Order to Supplier B = Action % 5` (Modulo)
*   **Range:** Both order quantities are integers from 0 to 4 (since `sqrt(25)-1 = 4`).

**Example:**
If the AI outputs the number **12**:
*   `12 // 5 = 2` -> Order **2 units** from Supplier A.
*   `12 % 5 = 2` -> Order **2 units** from Supplier B.

**Thesis Description:**
"The action space is discrete and multi-dimensional. Each agent creates a joint replenishment decision for its two upstream sources. The agent selects a single integer action $a_t \in [0, 24]$, which is decoded into two distinct order quantities $(q_1, q_2)$ where $q_1, q_2 \in \{0, 1, 2, 3, 4\}$, representing the volume to procure from each respective supplier."

## 3. The "Scoreboard" (Reward Function)

**Function:** `state_update(self, action_)` (Lines 283-354)

**The Formula:**
$$ R_t = - \left( \text{Ordering Cost} + \text{Holding Cost} + \text{Backlog Cost} \right) $$

Expanding the terms based on the code:
$$ R_t = - \left( (q_A \cdot C_A + q_B \cdot C_B) + (I_t \cdot H) + (B_t \cdot b) \right) $$

**Code Verification:**
Lines 330-331:
```python
b_c = B[i*2+j]*np.sum(self.backlog[(i)*2+j])
reward = - actual_order[0]*C[i][0] - actual_order[1]*C[i][1] - self.inventory[i][j]*H[i] - b_c
```

**Variable Default Values:**
*   **$C$ (Ordering Cost)**: `C = [[0.1, 0.2], ...]` (Line 15).
    *   Cost is 0.1 for Supplier A and 0.2 for Supplier B. This asymmetry forces the agent to learn to prefer the cheaper source (Supplier A) unless stockouts require using both.
*   **$H$ (Holding Cost)**: `H = [1, 1, ...]` (Line 16). Cost is 1.0 per unit per step.
*   **$b$ (Backlog Cost)**: `B = [1, 1, ...]` (Line 17). Cost is 1.0 per unit missed per step.

## 4. The "Daily Routine" (Step Logic)

**Function:** `state_update()`

**Warehouse Manager Analogy:**
Every "day" (time step), the warehouse manager performs this sequence:

1.  **Calculate Incoming Demand (Line 297-303):** 
    *   Retailers check customer demand.
    *   Wholesalers/Distributors check the orders placed by the retailers.
2.  **Receive Shipments (implicit vs explicit):**
    *   The code processes fulfillment *before* the new order is officially added to the "transit pipeline" for the *next* step, but effectively goods ordered `LEAD_TIME` ago arrive and are available for use in `E_S` (Effective Supply).
3.  **Fulfill Orders (Lines 313-317):**
    *   We determine `E_S` (Effective Supply): The maximum we can ship is `Current Inventory + Arriving Shipment`.
    *   We determine `N_S` (Normal Supply): How much of the demand we can satisfy.
4.  **Update Backlog (Lines 319-320):**
    *   If `Demand > Supply`, the difference is added to the Backlog.
5.  **Update Inventory (Line 323):**
    *   `Inventory = Old Inventory + Arriving Shipment - Shipped Goods`.
    *   (If we shipped everything we had, Inventory becomes 0).
6.  **Place New Orders (Line 325):**
    *   The decisions made by the AI (actions) are added to the order queue. These will arrive in 4 days.
7.  **Calculate Costs (Lines 330-331):**
    *   The manager tallies up the costs for holding stock, missing orders, and placing new orders to get the day's "Reward."

**Order of Events Thesis Summary:**
"In each time step, the simulation first aggregates demand from downstream entities. It then calculates the effective supply (available on-hand inventory plus arriving shipments). Demand is fulfilled to the extent possible; unmet demand is recorded as backlog, and remaining items are held as inventory. Finally, the agent's new replenishment orders are processed and entered into the deterministic lead-time pipeline."

## 5. Variable Glossary (Cheat Sheet)

| Variable | Definition | Role in Simulation |
| :--- | :--- | :--- |
| `self.inventory` | **Current Stock** | The number of units physically in the warehouse available for sale. |
| `self.backlog` | **Unmet Orders** | Cumulative total of orders received in the past that could not be filled due to stockouts. |
| `self.order` | **Transit Pipeline** | A list/queue of orders placed but not yet received. Represents goods "on the truck." |
| `LEAD_TIME` | **Delivery Delay** | Fixed at **4 days** (Line 21). An order placed at $t$ arrives at $t+4$. |
| `C` | **Ordering Cost** | Cost to purchase one unit. Asymmetric values (0.1 vs 0.2) simulate different supplier pricing. |
| `H` | **Holding Cost** | Cost to store one unit of inventory overnight (Value: 1.0). |
| `B` | **Penalty Cost** | Penalty for every unit of demand that is backlogged/missed (Value: 1.0). |
| `S_I` | **Start Inventory** | Initial inventory level at the start of an episode (Value: 2). |
