`timescale 1ns/1ps

//==============================================================================
// pe_int8: signed 8×8 multiply-accumulate, 32-bit accumulator
//==============================================================================
module pe_int8(
  input  wire           clk,
  input  wire           rst_n,
  input  wire signed [7:0]  a_in,
  input  wire signed [7:0]  b_in,
  input  wire signed [31:0] c_in,
  output reg  signed [7:0]  a_out,
  output reg  signed [7:0]  b_out,
  output reg  signed [31:0] c_out
);
  // 8×8→16 multiply, then accumulate into 32-bit
  wire signed [15:0] mult = a_in * b_in;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      a_out <= 8'sd0;
      b_out <= 8'sd0;
      c_out <= 32'sd0;
    end else begin
      a_out <= a_in;
      b_out <= b_in;
      c_out <= c_in + mult;
    end
  end
endmodule


//==============================================================================
// systolic_int8: 16×16 array of pe_int8
//==============================================================================
module systolic_int8 #(
  parameter N = 16
)(
  input  wire               clk,
  input  wire               rst_n,
  input  wire [8*N-1:0]     A_bus,   // packed A[0]…A[N-1], each signed [7:0]
  input  wire [8*N-1:0]     B_bus,   // packed B[0]…B[N-1]
  output wire [32*N*N-1:0]  C_bus    // packed C[0,0]…C[N-1,N-1], each signed [31:0]
);
  // unpack A and B
  wire signed [7:0] A_in [0:N-1];
  wire signed [7:0] B_in [0:N-1];
  genvar i,j;
  generate
    for (i=0; i<N; i=i+1) begin
      assign A_in[i] = A_bus[(i+1)*8-1 -: 8];
      assign B_in[i] = B_bus[(i+1)*8-1 -: 8];
    end
  endgenerate

  // internal pipelines
  wire signed [7:0]  a_pipe [0:N][0:N];
  wire signed [7:0]  b_pipe [0:N][0:N];
  wire signed [31:0] c_pipe [0:N][0:N];

  // boundary conditions
  generate
    for (i=0; i<N; i=i+1) begin
      assign a_pipe[i][0] = A_in[i];
      assign b_pipe[0][i] = B_in[i];
      assign c_pipe[i][0] = 32'sd0;
      assign c_pipe[0][i] = 32'sd0;
    end
  endgenerate

  // instantiate PEs
  generate
    for (i=0; i<N; i=i+1) begin: ROW
      for (j=0; j<N; j=j+1) begin: COL
        pe_int8 pe (
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

  // pack C outputs
  generate
    for (i=0; i<N; i=i+1) begin
      for (j=0; j<N; j=j+1) begin
        assign C_bus[((i*N + j + 1)*32-1) -: 32] = c_pipe[i+1][j+1];
      end
    end
  endgenerate
endmodule
