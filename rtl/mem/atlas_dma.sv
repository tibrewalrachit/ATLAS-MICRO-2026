//===========================================================================
// atlas_dma -- moves lines between HBDRAM and the core's scratchpad
//
// Weights dominate what an ATLAS core moves.  Qwen3-235B has 128 experts of
// 3 x 4096 x 1536 weights per layer, and at FP4 a single expert's up- and
// down-projections are still around 9 MB -- far past the 4 MB scratchpad -- so
// weight streaming, not reuse, sets the bandwidth the design needs.  That is
// the reason ATLAS bonds the DRAM to the logic die in the first place.
//
// The engine issues one line request per cycle while the controller accepts
// them, and the controller's FR-FCFS scheduler is what turns a linear
// descriptor into row hits.  Responses may return out of order, so each
// carries the buffer address it is destined for, tagged by request id.
//===========================================================================
`include "atlas_defs.svh"

module atlas_dma #(
  parameter int unsigned DQ_W    = ATLAS_HBD_DQ_W,   // 1024 bits per line
  parameter int unsigned ROW_W   = ATLAS_HBD_ROW_W,
  parameter int unsigned COL_W   = ATLAS_HBD_COL_W,
  parameter int unsigned ID_W    = 6,
  parameter int unsigned BUF_AW  = 20
) (
  input  logic                clk,
  input  logic                rst_n,

  // ---- descriptor ----
  input  logic                cmd_valid,
  output logic                cmd_ready,
  input  logic                cmd_store,        // 0 = DRAM->buffer, 1 = buffer->DRAM
  input  logic [ROW_W-1:0]    cmd_row,
  input  logic [COL_W-1:0]    cmd_col,
  input  logic [BUF_AW-1:0]   cmd_buf_addr,
  input  logic [15:0]         cmd_lines,

  output logic                busy,
  output logic                done,

  // ---- to the HBDRAM controller ----
  output logic                req_valid,
  input  logic                req_ready,
  output logic                req_we,
  output logic [ROW_W-1:0]    req_row,
  output logic [COL_W-1:0]    req_col,
  output logic [ID_W-1:0]     req_id,
  output logic [DQ_W-1:0]     req_wdata,
  input  logic                rsp_valid,
  input  logic [ID_W-1:0]     rsp_id,
  input  logic [DQ_W-1:0]     rsp_rdata,

  // ---- to the scratchpad ----
  output logic                buf_wr_en,
  output logic [BUF_AW-1:0]   buf_wr_addr,
  output logic [DQ_W-1:0]     buf_wr_data,
  output logic                buf_rd_en,
  output logic [BUF_AW-1:0]   buf_rd_addr,
  input  logic [DQ_W-1:0]     buf_rd_data
);

  localparam int unsigned NTAG = 1 << ID_W;

  typedef enum logic [1:0] { D_IDLE, D_RUN, D_WAIT } state_e;
  state_e state;

  logic [ROW_W-1:0]  cur_row;
  logic [COL_W-1:0]  cur_col;
  logic [BUF_AW-1:0] cur_buf;
  logic [15:0]       remaining, outstanding;
  logic              is_store;
  logic [ID_W-1:0]   next_id;

  // Where each outstanding read must land.  Responses come back out of order
  // whenever the scheduler favours a row hit, so the destination cannot be
  // inferred from arrival order.
  logic [BUF_AW-1:0] tag_addr [0:NTAG-1];

  assign cmd_ready = (state == D_IDLE);
  assign busy      = (state != D_IDLE);

  assign req_valid = (state == D_RUN) && (remaining != 16'd0);
  assign req_we    = is_store;
  assign req_row   = cur_row;
  assign req_col   = cur_col;
  assign req_id    = next_id;
  assign req_wdata = buf_rd_data;

  // For a store the line is fetched from the scratchpad a cycle ahead, so it
  // is on req_wdata when the request is accepted.
  assign buf_rd_en   = (state == D_RUN) && is_store;
  assign buf_rd_addr = cur_buf;

  assign buf_wr_en   = rsp_valid && !is_store;
  assign buf_wr_addr = tag_addr[rsp_id];
  assign buf_wr_data = rsp_rdata;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state       <= D_IDLE;
      done        <= 1'b0;
      remaining   <= 16'd0;
      outstanding <= 16'd0;
      next_id     <= '0;
    end else begin
      done <= 1'b0;

      case (state)
        D_IDLE: begin
          if (cmd_valid) begin
            cur_row     <= cmd_row;
            cur_col     <= cmd_col;
            cur_buf     <= cmd_buf_addr;
            remaining   <= cmd_lines;
            outstanding <= 16'd0;
            is_store    <= cmd_store;
            state       <= (cmd_lines == 16'd0) ? D_IDLE : D_RUN;
          end
        end

        D_RUN: begin
          if (req_valid && req_ready) begin
            tag_addr[next_id] <= cur_buf;
            next_id   <= next_id + 1'b1;
            cur_buf   <= cur_buf + 1'b1;
            remaining <= remaining - 1'b1;
            // Columns are walked before rows, which is what keeps a burst
            // inside one open row for as long as the row lasts.
            if (cur_col == {COL_W{1'b1}}) begin
              cur_col <= '0;
              cur_row <= cur_row + 1'b1;
            end else begin
              cur_col <= cur_col + 1'b1;
            end
            if (!is_store) outstanding <= outstanding + 1'b1;
            if (remaining == 16'd1) state <= D_WAIT;
          end
          if (rsp_valid && !is_store) outstanding <= outstanding - 1'b1;
        end

        D_WAIT: begin
          // Stores complete when issued; reads when their data has landed.
          if (rsp_valid && !is_store) outstanding <= outstanding - 1'b1;
          if (is_store || (outstanding == 16'd0) ||
              (rsp_valid && (outstanding == 16'd1))) begin
            done  <= 1'b1;
            state <= D_IDLE;
          end
        end

        default: state <= D_IDLE;
      endcase
    end
  end

endmodule
