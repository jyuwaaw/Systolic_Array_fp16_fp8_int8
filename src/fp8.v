`timescale 1ns/1ps

//==============================================================================
// fp8_mul: a × b → z, simple IEEE-like 8-bit float (no NaN/Inf handling)
//==============================================================================

module fp8_mul (
  input  wire [7:0] a,
  input  wire [7:0] b,
  output reg  [7:0] z
);
  // parameters
  localparam EXP_BIAS = 7;

  // unpack
  wire        sign_a = a[7], sign_b = b[7];
  wire [3:0]  exp_a  = a[6:3], exp_b  = b[6:3];
  wire [2:0]  man_a  = a[2:0],  man_b  = b[2:0];

  // compute product exponent + mantissa
  wire        sign_z = sign_a ^ sign_b;
  wire [4:0]  exp_sum = exp_a + exp_b - EXP_BIAS;               // 5 bits to catch overflow
  wire [3:0]  man_a_ext = {1'b1, man_a};                       // implicit 1
  wire [3:0]  man_b_ext = {1'b1, man_b};
  wire [7:0]  man_prod  = man_a_ext * man_b_ext;                // 4×4→8 bits

  // normalize: if MSB high, shift right and bump exp
  wire [3:0]  exp_norm = man_prod[7] ? exp_sum + 1 : exp_sum[3:0];
  wire [2:0]  man_norm = man_prod[7] ? man_prod[7:5] : man_prod[6:4];

  always @(*) begin
    z = { sign_z, exp_norm, man_norm };
  end
endmodule


//==============================================================================
// fp8_add:  c = a + b, same toy FP8 (no special cases)
//==============================================================================

module fp8_add (
  input  wire [7:0] a,
  input  wire [7:0] b,
  output reg  [7:0] z
);
  localparam EXP_BIAS = 7;

  // unpack
  wire        sa = a[7], sb = b[7];
  wire [3:0]  ea = a[6:3], eb = b[6:3];
  wire [2:0]  ma = a[2:0], mb = b[2:0];

  // align mantissas to common exponent
  wire [3:0]  de  = (ea > eb) ? (ea - eb) : (eb - ea);
  wire [3:0]  emx = (ea > eb) ? ea : eb;
  wire        sign_out = (ea > eb) ? sa : sb;

  // extend mantissas to 7 bits with guard bits
  wire [6:0] ma_ext = {1'b1, ma,   3'b000}; // [6]=1, [5:3]=man, [2:0]=0
  wire [6:0] mb_ext = {1'b1, mb,   3'b000};
  wire [6:0] mb_sh  = mb_ext >> de;

  // add/sub mantissas (we ignore opposite-sign subtraction for simplicity)
  wire [7:0] sum_m  = ma_ext + mb_sh;

  // normalize result
  wire [3:0] exp_n  = sum_m[7] ? emx + 1 : emx;
  wire [2:0] man_n  = sum_m[7] ? sum_m[7:5] : sum_m[6:4];

  always @(*) begin
    z = { sign_out, exp_n, man_n };
  end
endmodule


//==============================================================================
// 2) FP8 PE with parameterized accumulator width
//    (uses the fp8_mul/fp8_add from before)
//==============================================================================
module pe_fp8 #(
  parameter ACC_W = 16
)(
  input  wire                clk,
  input  wire                rst_n,
  input  wire        [7:0]   a_in,
  input  wire        [7:0]   b_in,
  input  wire [ACC_W-1:0]     c_in,    // lower 8 bits hold the FP8 sum
  output reg   [7:0]          a_out,
  output reg   [7:0]          b_out,
  output reg   [ACC_W-1:0]    c_out
);
  wire [7:0] mul8;
  wire [7:0] sum8;

  // your pure‑RTL FP8 multiply & add
  fp8_mul u_mul (.a(a_in), .b(b_in), .z(mul8));
  fp8_add u_add (.a(mul8), .b(c_in[7:0]), .z(sum8));

  // sign‑extend the 8‑bit FP8 result into ACC_W
  wire [ACC_W-1:0] sum_ext = {{(ACC_W-8){sum8[7]}}, sum8};

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      a_out <= 0; 
      b_out <= 0; 
      c_out <= 0;
    end else begin
      a_out <= a_in;
      b_out <= b_in;
      c_out <= sum_ext;            // ACC_W‑bit register
    end
  end
endmodule


//==============================================================================
// 16×16 Systolic Array using pe_fp8
//==============================================================================

module systolic_fp8 #(
  parameter N = 16
)(
  input  wire            clk,
  input  wire            rst_n,
  input  wire [8*N-1:0]  A_bus,
  input  wire [8*N-1:0]  B_bus,
  output wire [8*N*N-1:0] C_bus
);
  // unpack inputs
  wire [7:0] A_in [0:N-1];
  wire [7:0] B_in [0:N-1];
  genvar i,j;
  generate
    for (i=0; i<N; i=i+1) begin
      assign A_in[i] = A_bus[(i+1)*8-1 -: 8];
      assign B_in[i] = B_bus[(i+1)*8-1 -: 8];
    end
  endgenerate

  // internal pipes
  wire [7:0] a_pipe [0:N][0:N];
  wire [7:0] b_pipe [0:N][0:N];
  wire [7:0] c_pipe [0:N][0:N];

  // boundary
  generate
    for (i=0; i<N; i=i+1) begin
      assign a_pipe[i][0] = A_in[i];
      assign b_pipe[0][i] = B_in[i];
      assign c_pipe[i][0] = 8'd0;
      assign c_pipe[0][i] = 8'd0;
    end
  endgenerate

  // PEs
  generate
    for (i=0; i<N; i=i+1) begin: ROW
      for (j=0; j<N; j=j+1) begin: COL
        pe_fp8 pe (
          .clk   (clk),
          .rst_n (rst_n),
          .a_in  (a_pipe[i][j]),
          .b_in  (b_pipe[i][j]),
          .c_in  (c_pipe[i][j]),
          .a_out (a_pipe[i][j+1]),
          .b_out (b_pipe[i+1][j]),
          .c_out (c_pipe[i+1][j+1])
        );
      end
    end
  endgenerate

  // pack outputs
  generate
    for (i=0; i<N; i=i+1) begin
      for (j=0; j<N; j=j+1) begin
        assign C_bus[((i*N+j+1)*8-1) -: 8] = c_pipe[i+1][j+1];
      end
    end
  endgenerate
endmodule
