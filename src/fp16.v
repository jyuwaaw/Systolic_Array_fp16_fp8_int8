`timescale 1ns/1ps

//==============================================================================
// fp16_mul: a × b → z, 1-5-10 format, no special cases
//==============================================================================
module fp16_mul (
  input  wire [15:0] a,
  input  wire [15:0] b,
  output reg  [15:0] z
);
  localparam EXP_BIAS = 15;

  // unpack
  wire        sign_a = a[15];
  wire        sign_b = b[15];
  wire [4:0]  exp_a  = a[14:10];
  wire [4:0]  exp_b  = b[14:10];
  wire [9:0]  man_a  = a[9:0];
  wire [9:0]  man_b  = b[9:0];

  // extend mantissas (implicit 1)
  wire [10:0] man_a_ext = {1'b1, man_a};
  wire [10:0] man_b_ext = {1'b1, man_b};

  // multiply mantissas → 11×11→22 bits
  wire [21:0] man_prod  = man_a_ext * man_b_ext;

  // exponent sum minus bias
  wire [6:0]  exp_sum   = exp_a + exp_b - EXP_BIAS;  // needs 7 bits to catch overflow

  // normalize: if product MSB=1 at bit21, shift right & bump exp
  wire [4:0]  exp_norm  = man_prod[21] ? exp_sum[5:1] + 1 : exp_sum[4:0];
  wire [9:0]  man_norm  = man_prod[21]
                         ? man_prod[21:12]   // top 10 bits after shift
                         : man_prod[20:11];  // top 10 bits without shift

  // sign
  wire        sign_z    = sign_a ^ sign_b;

  always @(*) begin
    z = { sign_z, exp_norm, man_norm };
  end
endmodule


//==============================================================================
// fp16_add: z = a + b, same toy FP16, no special-case handling
//==============================================================================
module fp16_add (
  input  wire [15:0] a,
  input  wire [15:0] b,
  output reg  [15:0] z
);
  localparam EXP_BIAS = 15;

  // unpack
  wire        sa = a[15], sb = b[15];
  wire [4:0]  ea = a[14:10], eb = b[14:10];
  wire [9:0]  ma = a[9:0],  mb = b[9:0];

  // delta exponent & select max
  wire [4:0]  de     = (ea > eb) ? ea - eb : eb - ea;
  wire [4:0]  e_max  = (ea > eb) ? ea : eb;
  wire        s_out  = (ea > eb) ? sa : sb;

  // extend mantissas with 4 guard bits (1+10+4 = 15 bits)
  wire [14:0] ma_ext = {1'b1, ma, 4'b0000};
  wire [14:0] mb_ext = {1'b1, mb, 4'b0000};

  // shift smaller mantissa
  wire [14:0] mb_sh  = (ea > eb) ? (mb_ext >> de) : (ma_ext >> de);
  wire [14:0] ma_sh  = (ea > eb) ? ma_ext : mb_ext;

  // do the add (we ignore opposite-sign subtraction complexity)
  wire [15:0] sum_m  = ma_sh + mb_sh;

  // normalize sum
  wire [4:0]  exp_n  = sum_m[15] ? e_max + 1 : e_max;
  wire [9:0]  man_n  = sum_m[15]
                       ? sum_m[15:6]  // top 10 bits if carry-out
                       : sum_m[14:5]; // top 10 bits otherwise

  always @(*) begin
    z = { s_out, exp_n, man_n };
  end
endmodule


//==============================================================================
// 3) FP16 PE with parameterized accumulator width
//    (uses the fp16_mul/fp16_add from before)
//==============================================================================
module pe_fp16 #(
  parameter ACC_W = 16
)(
  input  wire                 clk,
  input  wire                 rst_n,
  input  wire        [15:0]   a_in,
  input  wire        [15:0]   b_in,
  input  wire [ACC_W-1:0]     c_in,
  output reg   [15:0]         a_out,
  output reg   [15:0]         b_out,
  output reg   [ACC_W-1:0]    c_out
);
  wire [15:0] mul16;
  wire [15:0] sum16;

  fp16_mul u_mul (.a(a_in), .b(b_in), .z(mul16));
  fp16_add u_add (.a(mul16), .b(c_in[15:0]), .z(sum16));

  // sign‑extend the 16‑bit result into ACC_W
  wire [ACC_W-1:0] sum_ext = {{(ACC_W-16){sum16[15]}}, sum16};

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      a_out <= 0; 
      b_out <= 0; 
      c_out <= 0;
    end else begin
      a_out <= a_in;
      b_out <= b_in;
      c_out <= sum_ext;
    end
  end
endmodule


//==============================================================================
// 16×16 Systolic Array using pe_fp16
//==============================================================================
module systolic_fp16 #(
  parameter N = 16
)(
  input  wire             clk,
  input  wire             rst_n,
  input  wire [16*N-1:0]  A_bus,  // A[0]…A[15] packed, each 16 bits
  input  wire [16*N-1:0]  B_bus,  // B[0]…B[15]
  output wire [(16*N*N)-1:0] C_bus // C[0,0]…C[15,15]
);
  // unpack inputs
  wire [15:0] A_in [0:N-1];
  wire [15:0] B_in [0:N-1];
  genvar i,j;
  generate
    for (i=0; i<N; i=i+1) begin
      assign A_in[i] = A_bus[(i+1)*16-1 -: 16];
      assign B_in[i] = B_bus[(i+1)*16-1 -: 16];
    end
  endgenerate

  // pipeline registers
  wire [15:0] a_pipe [0:N][0:N];
  wire [15:0] b_pipe [0:N][0:N];
  wire [15:0] c_pipe [0:N][0:N];

  // boundary conditions
  generate
    for (i=0; i<N; i=i+1) begin
      assign a_pipe[i][0] = A_in[i];
      assign b_pipe[0][i] = B_in[i];
      assign c_pipe[i][0] = 16'd0;
      assign c_pipe[0][i] = 16'd0;
    end
  endgenerate

  // instantiate PEs
  generate
    for (i=0; i<N; i=i+1) begin: ROW
      for (j=0; j<N; j=j+1) begin: COL
        pe_fp16 pe (
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
        assign C_bus[((i*N + j + 1)*16-1) -: 16] = c_pipe[i+1][j+1];
      end
    end
  endgenerate
endmodule
