��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
y

SegmentSum	
data"T
segment_ids"Tindices
output"T" 
Ttype:
2	"
Tindicestype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
SparseTensorDenseMatMul
	a_indices"Tindices
a_values"T
a_shape	
b"T
product"T"	
Ttype"
Tindicestype0	:
2	"
	adjoint_abool( "
	adjoint_bbool( 
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.11.12unknown8��
x
net/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namenet/dense_1/bias
q
$net/dense_1/bias/Read/ReadVariableOpReadVariableOpnet/dense_1/bias*
_output_shapes
:*
dtype0
�
net/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*#
shared_namenet/dense_1/kernel
y
&net/dense_1/kernel/Read/ReadVariableOpReadVariableOpnet/dense_1/kernel*
_output_shapes

:@*
dtype0
t
net/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namenet/dense/bias
m
"net/dense/bias/Read/ReadVariableOpReadVariableOpnet/dense/bias*
_output_shapes
:@*
dtype0
|
net/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*!
shared_namenet/dense/kernel
u
$net/dense/kernel/Read/ReadVariableOpReadVariableOpnet/dense/kernel*
_output_shapes

: @*
dtype0
�
)net/global_attention_pool/attn_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)net/global_attention_pool/attn_layer/bias
�
=net/global_attention_pool/attn_layer/bias/Read/ReadVariableOpReadVariableOp)net/global_attention_pool/attn_layer/bias*
_output_shapes
: *
dtype0
�
+net/global_attention_pool/attn_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *<
shared_name-+net/global_attention_pool/attn_layer/kernel
�
?net/global_attention_pool/attn_layer/kernel/Read/ReadVariableOpReadVariableOp+net/global_attention_pool/attn_layer/kernel*
_output_shapes

:@ *
dtype0
�
-net/global_attention_pool/features_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-net/global_attention_pool/features_layer/bias
�
Anet/global_attention_pool/features_layer/bias/Read/ReadVariableOpReadVariableOp-net/global_attention_pool/features_layer/bias*
_output_shapes
: *
dtype0
�
/net/global_attention_pool/features_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *@
shared_name1/net/global_attention_pool/features_layer/kernel
�
Cnet/global_attention_pool/features_layer/kernel/Read/ReadVariableOpReadVariableOp/net/global_attention_pool/features_layer/kernel*
_output_shapes

:@ *
dtype0
�
)net/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)net/batch_normalization_2/moving_variance
�
=net/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp)net/batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0
�
%net/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%net/batch_normalization_2/moving_mean
�
9net/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp%net/batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
�
net/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name net/batch_normalization_2/beta
�
2net/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpnet/batch_normalization_2/beta*
_output_shapes
:@*
dtype0
�
net/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!net/batch_normalization_2/gamma
�
3net/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpnet/batch_normalization_2/gamma*
_output_shapes
:@*
dtype0
~
net/gcn_conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_namenet/gcn_conv_1/bias
w
'net/gcn_conv_1/bias/Read/ReadVariableOpReadVariableOpnet/gcn_conv_1/bias*
_output_shapes
:@*
dtype0
�
net/gcn_conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_namenet/gcn_conv_1/kernel

)net/gcn_conv_1/kernel/Read/ReadVariableOpReadVariableOpnet/gcn_conv_1/kernel*
_output_shapes

:@@*
dtype0
�
)net/batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)net/batch_normalization_1/moving_variance
�
=net/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp)net/batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0
�
%net/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%net/batch_normalization_1/moving_mean
�
9net/batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp%net/batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0
�
net/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name net/batch_normalization_1/beta
�
2net/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpnet/batch_normalization_1/beta*
_output_shapes
:@*
dtype0
�
net/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!net/batch_normalization_1/gamma
�
3net/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpnet/batch_normalization_1/gamma*
_output_shapes
:@*
dtype0
z
net/gcn_conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_namenet/gcn_conv/bias
s
%net/gcn_conv/bias/Read/ReadVariableOpReadVariableOpnet/gcn_conv/bias*
_output_shapes
:@*
dtype0
�
net/gcn_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*$
shared_namenet/gcn_conv/kernel
{
'net/gcn_conv/kernel/Read/ReadVariableOpReadVariableOpnet/gcn_conv/kernel*
_output_shapes

:@*
dtype0
�
'net/batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'net/batch_normalization/moving_variance
�
;net/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp'net/batch_normalization/moving_variance*
_output_shapes
:*
dtype0
�
#net/batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#net/batch_normalization/moving_mean
�
7net/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp#net/batch_normalization/moving_mean*
_output_shapes
:*
dtype0
�
net/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namenet/batch_normalization/beta
�
0net/batch_normalization/beta/Read/ReadVariableOpReadVariableOpnet/batch_normalization/beta*
_output_shapes
:*
dtype0
�
net/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namenet/batch_normalization/gamma
�
1net/batch_normalization/gamma/Read/ReadVariableOpReadVariableOpnet/batch_normalization/gamma*
_output_shapes
:*
dtype0
y
serving_default_args_0Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
{
serving_default_args_0_1Placeholder*'
_output_shapes
:���������*
dtype0	*
shape:���������
s
serving_default_args_0_2Placeholder*#
_output_shapes
:���������*
dtype0*
shape:���������
a
serving_default_args_0_3Placeholder*
_output_shapes
:*
dtype0	*
shape:
s
serving_default_args_0_4Placeholder*#
_output_shapes
:���������*
dtype0	*
shape:���������
�	
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0serving_default_args_0_1serving_default_args_0_2serving_default_args_0_3serving_default_args_0_4#net/batch_normalization/moving_mean'net/batch_normalization/moving_variancenet/batch_normalization/betanet/batch_normalization/gammanet/gcn_conv/kernelnet/gcn_conv/bias%net/batch_normalization_1/moving_mean)net/batch_normalization_1/moving_variancenet/batch_normalization_1/betanet/batch_normalization_1/gammanet/gcn_conv_1/kernelnet/gcn_conv_1/bias%net/batch_normalization_2/moving_mean)net/batch_normalization_2/moving_variancenet/batch_normalization_2/betanet/batch_normalization_2/gamma/net/global_attention_pool/features_layer/kernel-net/global_attention_pool/features_layer/bias+net/global_attention_pool/attn_layer/kernel)net/global_attention_pool/attn_layer/biasnet/dense/kernelnet/dense/biasnet/dense_1/kernelnet/dense_1/bias*(
Tin!
2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_2989171

NoOpNoOp
�J
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�J
value�JB�J B�J
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

batchnorm1
		conv1


batchnorm2
	conv2

batchnorm3
global_pool

dense1

dense2

signatures*
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
 15
!16
"17
#18
$19
%20
&21
'22
(23*
�
0
1
2
3
4
5
6
7
8
9
!10
"11
#12
$13
%14
&15
'16
(17*
* 
�
)non_trainable_variables

*layers
+metrics
,layer_regularization_losses
-layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

.trace_0
/trace_1* 

0trace_0
1trace_1* 
* 
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses
8axis
	gamma
beta
moving_mean
moving_variance*
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?kwargs_keys

kernel
bias*
�
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses
Faxis
	gamma
beta
moving_mean
moving_variance*
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses
Mkwargs_keys

kernel
bias*
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
Taxis
	gamma
beta
moving_mean
 moving_variance*
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses
[features_layer
\attention_layer*
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses

%kernel
&bias*
�
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

'kernel
(bias*

iserving_default* 
]W
VARIABLE_VALUEnet/batch_normalization/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEnet/batch_normalization/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#net/batch_normalization/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'net/batch_normalization/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEnet/gcn_conv/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEnet/gcn_conv/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEnet/batch_normalization_1/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEnet/batch_normalization_1/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%net/batch_normalization_1/moving_mean&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE)net/batch_normalization_1/moving_variance&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEnet/gcn_conv_1/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEnet/gcn_conv_1/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEnet/batch_normalization_2/gamma'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEnet/batch_normalization_2/beta'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%net/batch_normalization_2/moving_mean'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE)net/batch_normalization_2/moving_variance'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE/net/global_attention_pool/features_layer/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE-net/global_attention_pool/features_layer/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE+net/global_attention_pool/attn_layer/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE)net/global_attention_pool/attn_layer/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEnet/dense/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEnet/dense/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEnet/dense_1/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEnet/dense_1/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
.
0
1
2
3
4
 5*
<
0
	1

2
3
4
5
6
7*
* 
* 
* 
* 
* 
* 
* 
 
0
1
2
3*

0
1*
* 
�
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*

otrace_0
ptrace_1* 

qtrace_0
rtrace_1* 
* 

0
1*

0
1*
* 
�
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*

xtrace_0* 

ytrace_0* 
* 
 
0
1
2
3*

0
1*
* 
�
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*

trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
 
0
1
2
 3*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
 
!0
"1
#2
$3*
 
!0
"1
#2
$3*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

!kernel
"bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

#kernel
$bias*

%0
&1*

%0
&1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

'0
(1*

'0
(1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
 1*
* 
* 
* 
* 
* 
* 
* 
* 
* 

[0
\1*
* 
* 
* 
* 
* 

!0
"1*

!0
"1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

#0
$1*

#0
$1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename1net/batch_normalization/gamma/Read/ReadVariableOp0net/batch_normalization/beta/Read/ReadVariableOp7net/batch_normalization/moving_mean/Read/ReadVariableOp;net/batch_normalization/moving_variance/Read/ReadVariableOp'net/gcn_conv/kernel/Read/ReadVariableOp%net/gcn_conv/bias/Read/ReadVariableOp3net/batch_normalization_1/gamma/Read/ReadVariableOp2net/batch_normalization_1/beta/Read/ReadVariableOp9net/batch_normalization_1/moving_mean/Read/ReadVariableOp=net/batch_normalization_1/moving_variance/Read/ReadVariableOp)net/gcn_conv_1/kernel/Read/ReadVariableOp'net/gcn_conv_1/bias/Read/ReadVariableOp3net/batch_normalization_2/gamma/Read/ReadVariableOp2net/batch_normalization_2/beta/Read/ReadVariableOp9net/batch_normalization_2/moving_mean/Read/ReadVariableOp=net/batch_normalization_2/moving_variance/Read/ReadVariableOpCnet/global_attention_pool/features_layer/kernel/Read/ReadVariableOpAnet/global_attention_pool/features_layer/bias/Read/ReadVariableOp?net/global_attention_pool/attn_layer/kernel/Read/ReadVariableOp=net/global_attention_pool/attn_layer/bias/Read/ReadVariableOp$net/dense/kernel/Read/ReadVariableOp"net/dense/bias/Read/ReadVariableOp&net/dense_1/kernel/Read/ReadVariableOp$net/dense_1/bias/Read/ReadVariableOpConst*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_2989993
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamenet/batch_normalization/gammanet/batch_normalization/beta#net/batch_normalization/moving_mean'net/batch_normalization/moving_variancenet/gcn_conv/kernelnet/gcn_conv/biasnet/batch_normalization_1/gammanet/batch_normalization_1/beta%net/batch_normalization_1/moving_mean)net/batch_normalization_1/moving_variancenet/gcn_conv_1/kernelnet/gcn_conv_1/biasnet/batch_normalization_2/gammanet/batch_normalization_2/beta%net/batch_normalization_2/moving_mean)net/batch_normalization_2/moving_variance/net/global_attention_pool/features_layer/kernel-net/global_attention_pool/features_layer/bias+net/global_attention_pool/attn_layer/kernel)net/global_attention_pool/attn_layer/biasnet/dense/kernelnet/dense/biasnet/dense_1/kernelnet/dense_1/bias*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_2990075��

�8
�
 __inference__traced_save_2989993
file_prefix<
8savev2_net_batch_normalization_gamma_read_readvariableop;
7savev2_net_batch_normalization_beta_read_readvariableopB
>savev2_net_batch_normalization_moving_mean_read_readvariableopF
Bsavev2_net_batch_normalization_moving_variance_read_readvariableop2
.savev2_net_gcn_conv_kernel_read_readvariableop0
,savev2_net_gcn_conv_bias_read_readvariableop>
:savev2_net_batch_normalization_1_gamma_read_readvariableop=
9savev2_net_batch_normalization_1_beta_read_readvariableopD
@savev2_net_batch_normalization_1_moving_mean_read_readvariableopH
Dsavev2_net_batch_normalization_1_moving_variance_read_readvariableop4
0savev2_net_gcn_conv_1_kernel_read_readvariableop2
.savev2_net_gcn_conv_1_bias_read_readvariableop>
:savev2_net_batch_normalization_2_gamma_read_readvariableop=
9savev2_net_batch_normalization_2_beta_read_readvariableopD
@savev2_net_batch_normalization_2_moving_mean_read_readvariableopH
Dsavev2_net_batch_normalization_2_moving_variance_read_readvariableopN
Jsavev2_net_global_attention_pool_features_layer_kernel_read_readvariableopL
Hsavev2_net_global_attention_pool_features_layer_bias_read_readvariableopJ
Fsavev2_net_global_attention_pool_attn_layer_kernel_read_readvariableopH
Dsavev2_net_global_attention_pool_attn_layer_bias_read_readvariableop/
+savev2_net_dense_kernel_read_readvariableop-
)savev2_net_dense_bias_read_readvariableop1
-savev2_net_dense_1_kernel_read_readvariableop/
+savev2_net_dense_1_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:08savev2_net_batch_normalization_gamma_read_readvariableop7savev2_net_batch_normalization_beta_read_readvariableop>savev2_net_batch_normalization_moving_mean_read_readvariableopBsavev2_net_batch_normalization_moving_variance_read_readvariableop.savev2_net_gcn_conv_kernel_read_readvariableop,savev2_net_gcn_conv_bias_read_readvariableop:savev2_net_batch_normalization_1_gamma_read_readvariableop9savev2_net_batch_normalization_1_beta_read_readvariableop@savev2_net_batch_normalization_1_moving_mean_read_readvariableopDsavev2_net_batch_normalization_1_moving_variance_read_readvariableop0savev2_net_gcn_conv_1_kernel_read_readvariableop.savev2_net_gcn_conv_1_bias_read_readvariableop:savev2_net_batch_normalization_2_gamma_read_readvariableop9savev2_net_batch_normalization_2_beta_read_readvariableop@savev2_net_batch_normalization_2_moving_mean_read_readvariableopDsavev2_net_batch_normalization_2_moving_variance_read_readvariableopJsavev2_net_global_attention_pool_features_layer_kernel_read_readvariableopHsavev2_net_global_attention_pool_features_layer_bias_read_readvariableopFsavev2_net_global_attention_pool_attn_layer_kernel_read_readvariableopDsavev2_net_global_attention_pool_attn_layer_bias_read_readvariableop+savev2_net_dense_kernel_read_readvariableop)savev2_net_dense_bias_read_readvariableop-savev2_net_dense_1_kernel_read_readvariableop+savev2_net_dense_1_bias_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *'
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :::::@:@:@:@:@:@:@@:@:@:@:@:@:@ : :@ : : @:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: @: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: 
�j
�
#__inference__traced_restore_2990075
file_prefix<
.assignvariableop_net_batch_normalization_gamma:=
/assignvariableop_1_net_batch_normalization_beta:D
6assignvariableop_2_net_batch_normalization_moving_mean:H
:assignvariableop_3_net_batch_normalization_moving_variance:8
&assignvariableop_4_net_gcn_conv_kernel:@2
$assignvariableop_5_net_gcn_conv_bias:@@
2assignvariableop_6_net_batch_normalization_1_gamma:@?
1assignvariableop_7_net_batch_normalization_1_beta:@F
8assignvariableop_8_net_batch_normalization_1_moving_mean:@J
<assignvariableop_9_net_batch_normalization_1_moving_variance:@;
)assignvariableop_10_net_gcn_conv_1_kernel:@@5
'assignvariableop_11_net_gcn_conv_1_bias:@A
3assignvariableop_12_net_batch_normalization_2_gamma:@@
2assignvariableop_13_net_batch_normalization_2_beta:@G
9assignvariableop_14_net_batch_normalization_2_moving_mean:@K
=assignvariableop_15_net_batch_normalization_2_moving_variance:@U
Cassignvariableop_16_net_global_attention_pool_features_layer_kernel:@ O
Aassignvariableop_17_net_global_attention_pool_features_layer_bias: Q
?assignvariableop_18_net_global_attention_pool_attn_layer_kernel:@ K
=assignvariableop_19_net_global_attention_pool_attn_layer_bias: 6
$assignvariableop_20_net_dense_kernel: @0
"assignvariableop_21_net_dense_bias:@8
&assignvariableop_22_net_dense_1_kernel:@2
$assignvariableop_23_net_dense_1_bias:
identity_25��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp.assignvariableop_net_batch_normalization_gammaIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp/assignvariableop_1_net_batch_normalization_betaIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp6assignvariableop_2_net_batch_normalization_moving_meanIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp:assignvariableop_3_net_batch_normalization_moving_varianceIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp&assignvariableop_4_net_gcn_conv_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp$assignvariableop_5_net_gcn_conv_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp2assignvariableop_6_net_batch_normalization_1_gammaIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp1assignvariableop_7_net_batch_normalization_1_betaIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp8assignvariableop_8_net_batch_normalization_1_moving_meanIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp<assignvariableop_9_net_batch_normalization_1_moving_varianceIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp)assignvariableop_10_net_gcn_conv_1_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp'assignvariableop_11_net_gcn_conv_1_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp3assignvariableop_12_net_batch_normalization_2_gammaIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp2assignvariableop_13_net_batch_normalization_2_betaIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp9assignvariableop_14_net_batch_normalization_2_moving_meanIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp=assignvariableop_15_net_batch_normalization_2_moving_varianceIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpCassignvariableop_16_net_global_attention_pool_features_layer_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpAassignvariableop_17_net_global_attention_pool_features_layer_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp?assignvariableop_18_net_global_attention_pool_attn_layer_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp=assignvariableop_19_net_global_attention_pool_attn_layer_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp$assignvariableop_20_net_dense_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp"assignvariableop_21_net_dense_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp&assignvariableop_22_net_dense_1_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp$assignvariableop_23_net_dense_1_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_25Identity_25:output:0*E
_input_shapes4
2: : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2989680

inputs*
cast_readvariableop_resource:@,
cast_1_readvariableop_resource:@,
cast_2_readvariableop_resource:@,
cast_3_readvariableop_resource:@
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_1_layer_call_fn_2989660

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2988638o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2988591

inputs*
cast_readvariableop_resource:@,
cast_1_readvariableop_resource:@,
cast_2_readvariableop_resource:@,
cast_3_readvariableop_resource:@
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_2_layer_call_fn_2989754

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2988673o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
B__inference_dense_layer_call_and_return_conditional_losses_2988856

inputs0
matmul_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�7
�
@__inference_net_layer_call_and_return_conditional_losses_2988879

inputs
inputs_1	
inputs_2
inputs_3	
inputs_4	)
batch_normalization_2988746:)
batch_normalization_2988748:)
batch_normalization_2988750:)
batch_normalization_2988752:"
gcn_conv_2988771:@
gcn_conv_2988773:@+
batch_normalization_1_2988776:@+
batch_normalization_1_2988778:@+
batch_normalization_1_2988780:@+
batch_normalization_1_2988782:@$
gcn_conv_1_2988801:@@ 
gcn_conv_1_2988803:@+
batch_normalization_2_2988806:@+
batch_normalization_2_2988808:@+
batch_normalization_2_2988810:@+
batch_normalization_2_2988812:@/
global_attention_pool_2988836:@ +
global_attention_pool_2988838: /
global_attention_pool_2988840:@ +
global_attention_pool_2988842: 
dense_2988857: @
dense_2988859:@!
dense_1_2988873:@
dense_1_2988875:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall� gcn_conv/StatefulPartitionedCall�"gcn_conv_1/StatefulPartitionedCall�-global_attention_pool/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_2988746batch_normalization_2988748batch_normalization_2988750batch_normalization_2988752*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2988509�
 gcn_conv/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0inputs_1inputs_2inputs_3gcn_conv_2988771gcn_conv_2988773*
Tin

2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_gcn_conv_layer_call_and_return_conditional_losses_2988770�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)gcn_conv/StatefulPartitionedCall:output:0batch_normalization_1_2988776batch_normalization_1_2988778batch_normalization_1_2988780batch_normalization_1_2988782*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2988591�
"gcn_conv_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0inputs_1inputs_2inputs_3gcn_conv_1_2988801gcn_conv_1_2988803*
Tin

2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_gcn_conv_1_layer_call_and_return_conditional_losses_2988800�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall+gcn_conv_1/StatefulPartitionedCall:output:0batch_normalization_2_2988806batch_normalization_2_2988808batch_normalization_2_2988810batch_normalization_2_2988812*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2988673�
-global_attention_pool/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0inputs_4global_attention_pool_2988836global_attention_pool_2988838global_attention_pool_2988840global_attention_pool_2988842*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_global_attention_pool_layer_call_and_return_conditional_losses_2988835�
dense/StatefulPartitionedCallStatefulPartitionedCall6global_attention_pool/StatefulPartitionedCall:output:0dense_2988857dense_2988859*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_2988856�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_2988873dense_1_2988875*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_2988872w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall!^gcn_conv/StatefulPartitionedCall#^gcn_conv_1/StatefulPartitionedCall.^global_attention_pool/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes|
z:���������:���������:���������::���������: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 gcn_conv/StatefulPartitionedCall gcn_conv/StatefulPartitionedCall2H
"gcn_conv_1/StatefulPartitionedCall"gcn_conv_1/StatefulPartitionedCall2^
-global_attention_pool/StatefulPartitionedCall-global_attention_pool/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2988509

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
%__inference_net_layer_call_fn_2989285
inputs_0

inputs	
inputs_1
inputs_2	

inputs_2_1	
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@ 

unknown_16: 

unknown_17:@ 

unknown_18: 

unknown_19: @

unknown_20:@

unknown_21:@

unknown_22:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2
inputs_2_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*(
Tin!
2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_net_layer_call_and_return_conditional_losses_2989061o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes|
z:���������:���������:���������::���������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs:MI
#
_output_shapes
:���������
"
_user_specified_name
inputs_2
�
�
R__inference_global_attention_pool_layer_call_and_return_conditional_losses_2989855
inputs_0
inputs_1	?
-features_layer_matmul_readvariableop_resource:@ <
.features_layer_biasadd_readvariableop_resource: ;
)attn_layer_matmul_readvariableop_resource:@ 8
*attn_layer_biasadd_readvariableop_resource: 
identity��!attn_layer/BiasAdd/ReadVariableOp� attn_layer/MatMul/ReadVariableOp�%features_layer/BiasAdd/ReadVariableOp�$features_layer/MatMul/ReadVariableOp�
$features_layer/MatMul/ReadVariableOpReadVariableOp-features_layer_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
features_layer/MatMulMatMulinputs_0,features_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
%features_layer/BiasAdd/ReadVariableOpReadVariableOp.features_layer_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
features_layer/BiasAddBiasAddfeatures_layer/MatMul:product:0-features_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
 attn_layer/MatMul/ReadVariableOpReadVariableOp)attn_layer_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
attn_layer/MatMulMatMulinputs_0(attn_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
!attn_layer/BiasAdd/ReadVariableOpReadVariableOp*attn_layer_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
attn_layer/BiasAddBiasAddattn_layer/MatMul:product:0)attn_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� l
attn_layer/SigmoidSigmoidattn_layer/BiasAdd:output:0*
T0*'
_output_shapes
:��������� u
mulMulfeatures_layer/BiasAdd:output:0attn_layer/Sigmoid:y:0*
T0*'
_output_shapes
:��������� m

SegmentSum
SegmentSummul:z:0inputs_1*
T0*
Tindices0	*'
_output_shapes
:��������� b
IdentityIdentitySegmentSum:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp"^attn_layer/BiasAdd/ReadVariableOp!^attn_layer/MatMul/ReadVariableOp&^features_layer/BiasAdd/ReadVariableOp%^features_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:���������@:���������: : : : 2F
!attn_layer/BiasAdd/ReadVariableOp!attn_layer/BiasAdd/ReadVariableOp2D
 attn_layer/MatMul/ReadVariableOp attn_layer/MatMul/ReadVariableOp2N
%features_layer/BiasAdd/ReadVariableOp%features_layer/BiasAdd/ReadVariableOp2L
$features_layer/MatMul/ReadVariableOp$features_layer/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������@
"
_user_specified_name
inputs_0:MI
#
_output_shapes
:���������
"
_user_specified_name
inputs_1
�$
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2988638

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@*
cast_readvariableop_resource:@,
cast_1_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
B__inference_dense_layer_call_and_return_conditional_losses_2989875

inputs0
matmul_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
*__inference_gcn_conv_layer_call_fn_2989619
inputs_0

inputs	
inputs_1
inputs_2	
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2unknown	unknown_0*
Tin

2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_gcn_conv_layer_call_and_return_conditional_losses_2988770o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������:: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�	
�
D__inference_dense_1_layer_call_and_return_conditional_losses_2989894

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�$
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2989821

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@*
cast_readvariableop_resource:@,
cast_1_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
E__inference_gcn_conv_layer_call_and_return_conditional_losses_2989634
inputs_0

inputs	
inputs_1
inputs_2	0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0k
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2MatMul:product:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAdd9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2988673

inputs*
cast_readvariableop_resource:@,
cast_1_readvariableop_resource:@,
cast_2_readvariableop_resource:@,
cast_3_readvariableop_resource:@
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_1_layer_call_fn_2989647

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2988591o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
D__inference_dense_1_layer_call_and_return_conditional_losses_2988872

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
G__inference_gcn_conv_1_layer_call_and_return_conditional_losses_2989741
inputs_0

inputs	
inputs_1
inputs_2	0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0k
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2MatMul:product:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAdd9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������@:���������:���������:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
'
_output_shapes
:���������@
"
_user_specified_name
inputs_0:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_2_layer_call_fn_2989767

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2988720o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
'__inference_dense_layer_call_fn_2989864

inputs
unknown: @
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_2988856o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
7__inference_global_attention_pool_layer_call_fn_2989835
inputs_0
inputs_1	
unknown:@ 
	unknown_0: 
	unknown_1:@ 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_global_attention_pool_layer_call_and_return_conditional_losses_2988835o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:���������@:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������@
"
_user_specified_name
inputs_0:MI
#
_output_shapes
:���������
"
_user_specified_name
inputs_1
�
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2989787

inputs*
cast_readvariableop_resource:@,
cast_1_readvariableop_resource:@,
cast_2_readvariableop_resource:@,
cast_3_readvariableop_resource:@
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
��
�
@__inference_net_layer_call_and_return_conditional_losses_2989527
inputs_0

inputs	
inputs_1
inputs_2	

inputs_2_1	I
;batch_normalization_assignmovingavg_readvariableop_resource:K
=batch_normalization_assignmovingavg_1_readvariableop_resource:>
0batch_normalization_cast_readvariableop_resource:@
2batch_normalization_cast_1_readvariableop_resource:9
'gcn_conv_matmul_readvariableop_resource:@6
(gcn_conv_biasadd_readvariableop_resource:@K
=batch_normalization_1_assignmovingavg_readvariableop_resource:@M
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:@@
2batch_normalization_1_cast_readvariableop_resource:@B
4batch_normalization_1_cast_1_readvariableop_resource:@;
)gcn_conv_1_matmul_readvariableop_resource:@@8
*gcn_conv_1_biasadd_readvariableop_resource:@K
=batch_normalization_2_assignmovingavg_readvariableop_resource:@M
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:@@
2batch_normalization_2_cast_readvariableop_resource:@B
4batch_normalization_2_cast_1_readvariableop_resource:@U
Cglobal_attention_pool_features_layer_matmul_readvariableop_resource:@ R
Dglobal_attention_pool_features_layer_biasadd_readvariableop_resource: Q
?global_attention_pool_attn_layer_matmul_readvariableop_resource:@ N
@global_attention_pool_attn_layer_biasadd_readvariableop_resource: 6
$dense_matmul_readvariableop_resource: @3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:
identity��#batch_normalization/AssignMovingAvg�2batch_normalization/AssignMovingAvg/ReadVariableOp�%batch_normalization/AssignMovingAvg_1�4batch_normalization/AssignMovingAvg_1/ReadVariableOp�'batch_normalization/Cast/ReadVariableOp�)batch_normalization/Cast_1/ReadVariableOp�%batch_normalization_1/AssignMovingAvg�4batch_normalization_1/AssignMovingAvg/ReadVariableOp�'batch_normalization_1/AssignMovingAvg_1�6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_1/Cast/ReadVariableOp�+batch_normalization_1/Cast_1/ReadVariableOp�%batch_normalization_2/AssignMovingAvg�4batch_normalization_2/AssignMovingAvg/ReadVariableOp�'batch_normalization_2/AssignMovingAvg_1�6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_2/Cast/ReadVariableOp�+batch_normalization_2/Cast_1/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�gcn_conv/BiasAdd/ReadVariableOp�gcn_conv/MatMul/ReadVariableOp�!gcn_conv_1/BiasAdd/ReadVariableOp� gcn_conv_1/MatMul/ReadVariableOp�7global_attention_pool/attn_layer/BiasAdd/ReadVariableOp�6global_attention_pool/attn_layer/MatMul/ReadVariableOp�;global_attention_pool/features_layer/BiasAdd/ReadVariableOp�:global_attention_pool/features_layer/MatMul/ReadVariableOp|
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
 batch_normalization/moments/meanMeaninputs_0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes

:�
-batch_normalization/moments/SquaredDifferenceSquaredDifferenceinputs_01batch_normalization/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 n
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:�
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0p
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0�
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:x
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:�
#batch_normalization/batchnorm/mul_1Mulinputs_0%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:�
!batch_normalization/batchnorm/subSub/batch_normalization/Cast/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
gcn_conv/MatMul/ReadVariableOpReadVariableOp'gcn_conv_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
gcn_conv/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0&gcn_conv/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
8gcn_conv/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2gcn_conv/MatMul:product:0*
T0*'
_output_shapes
:���������@�
gcn_conv/BiasAdd/ReadVariableOpReadVariableOp(gcn_conv_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
gcn_conv/BiasAddBiasAddBgcn_conv/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0'gcn_conv/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
gcn_conv/ReluRelugcn_conv/BiasAdd:output:0*
T0*'
_output_shapes
:���������@~
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_1/moments/meanMeangcn_conv/Relu:activations:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes

:@�
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencegcn_conv/Relu:activations:03batch_normalization_1/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������@�
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 �
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 p
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes
:@�
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@�
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0�
)batch_normalization_1/Cast/ReadVariableOpReadVariableOp2batch_normalization_1_cast_readvariableop_resource*
_output_shapes
:@*
dtype0�
+batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:03batch_normalization_1/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
%batch_normalization_1/batchnorm/mul_1Mulgcn_conv/Relu:activations:0'batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
#batch_normalization_1/batchnorm/subSub1batch_normalization_1/Cast/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
 gcn_conv_1/MatMul/ReadVariableOpReadVariableOp)gcn_conv_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
gcn_conv_1/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:0(gcn_conv_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
:gcn_conv_1/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2gcn_conv_1/MatMul:product:0*
T0*'
_output_shapes
:���������@�
!gcn_conv_1/BiasAdd/ReadVariableOpReadVariableOp*gcn_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
gcn_conv_1/BiasAddBiasAddDgcn_conv_1/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0)gcn_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@f
gcn_conv_1/ReluRelugcn_conv_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@~
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_2/moments/meanMeangcn_conv_1/Relu:activations:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

:@�
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencegcn_conv_1/Relu:activations:03batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������@�
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 �
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 p
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:@�
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@�
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0�
)batch_normalization_2/Cast/ReadVariableOpReadVariableOp2batch_normalization_2_cast_readvariableop_resource*
_output_shapes
:@*
dtype0�
+batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:03batch_normalization_2/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
%batch_normalization_2/batchnorm/mul_1Mulgcn_conv_1/Relu:activations:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
#batch_normalization_2/batchnorm/subSub1batch_normalization_2/Cast/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
:global_attention_pool/features_layer/MatMul/ReadVariableOpReadVariableOpCglobal_attention_pool_features_layer_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
+global_attention_pool/features_layer/MatMulMatMul)batch_normalization_2/batchnorm/add_1:z:0Bglobal_attention_pool/features_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
;global_attention_pool/features_layer/BiasAdd/ReadVariableOpReadVariableOpDglobal_attention_pool_features_layer_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
,global_attention_pool/features_layer/BiasAddBiasAdd5global_attention_pool/features_layer/MatMul:product:0Cglobal_attention_pool/features_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
6global_attention_pool/attn_layer/MatMul/ReadVariableOpReadVariableOp?global_attention_pool_attn_layer_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
'global_attention_pool/attn_layer/MatMulMatMul)batch_normalization_2/batchnorm/add_1:z:0>global_attention_pool/attn_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
7global_attention_pool/attn_layer/BiasAdd/ReadVariableOpReadVariableOp@global_attention_pool_attn_layer_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
(global_attention_pool/attn_layer/BiasAddBiasAdd1global_attention_pool/attn_layer/MatMul:product:0?global_attention_pool/attn_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
(global_attention_pool/attn_layer/SigmoidSigmoid1global_attention_pool/attn_layer/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
global_attention_pool/mulMul5global_attention_pool/features_layer/BiasAdd:output:0,global_attention_pool/attn_layer/Sigmoid:y:0*
T0*'
_output_shapes
:��������� �
 global_attention_pool/SegmentSum
SegmentSumglobal_attention_pool/mul:z:0
inputs_2_1*
T0*
Tindices0	*'
_output_shapes
:��������� �
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: @*
dtype0�
dense/MatMulMatMul)global_attention_pool/SegmentSum:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_2/Cast/ReadVariableOp,^batch_normalization_2/Cast_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp ^gcn_conv/BiasAdd/ReadVariableOp^gcn_conv/MatMul/ReadVariableOp"^gcn_conv_1/BiasAdd/ReadVariableOp!^gcn_conv_1/MatMul/ReadVariableOp8^global_attention_pool/attn_layer/BiasAdd/ReadVariableOp7^global_attention_pool/attn_layer/MatMul/ReadVariableOp<^global_attention_pool/features_layer/BiasAdd/ReadVariableOp;^global_attention_pool/features_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes|
z:���������:���������:���������::���������: : : : : : : : : : : : : : : : : : : : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2N
%batch_normalization_1/AssignMovingAvg%batch_normalization_1/AssignMovingAvg2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_1'batch_normalization_1/AssignMovingAvg_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_1/Cast/ReadVariableOp)batch_normalization_1/Cast/ReadVariableOp2Z
+batch_normalization_1/Cast_1/ReadVariableOp+batch_normalization_1/Cast_1/ReadVariableOp2N
%batch_normalization_2/AssignMovingAvg%batch_normalization_2/AssignMovingAvg2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_2/AssignMovingAvg_1'batch_normalization_2/AssignMovingAvg_12p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_2/Cast/ReadVariableOp)batch_normalization_2/Cast/ReadVariableOp2Z
+batch_normalization_2/Cast_1/ReadVariableOp+batch_normalization_2/Cast_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2B
gcn_conv/BiasAdd/ReadVariableOpgcn_conv/BiasAdd/ReadVariableOp2@
gcn_conv/MatMul/ReadVariableOpgcn_conv/MatMul/ReadVariableOp2F
!gcn_conv_1/BiasAdd/ReadVariableOp!gcn_conv_1/BiasAdd/ReadVariableOp2D
 gcn_conv_1/MatMul/ReadVariableOp gcn_conv_1/MatMul/ReadVariableOp2r
7global_attention_pool/attn_layer/BiasAdd/ReadVariableOp7global_attention_pool/attn_layer/BiasAdd/ReadVariableOp2p
6global_attention_pool/attn_layer/MatMul/ReadVariableOp6global_attention_pool/attn_layer/MatMul/ReadVariableOp2z
;global_attention_pool/features_layer/BiasAdd/ReadVariableOp;global_attention_pool/features_layer/BiasAdd/ReadVariableOp2x
:global_attention_pool/features_layer/MatMul/ReadVariableOp:global_attention_pool/features_layer/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs:MI
#
_output_shapes
:���������
"
_user_specified_name
inputs_2
�
�
5__inference_batch_normalization_layer_call_fn_2989540

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2988509o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
R__inference_global_attention_pool_layer_call_and_return_conditional_losses_2988835

inputs
inputs_1	?
-features_layer_matmul_readvariableop_resource:@ <
.features_layer_biasadd_readvariableop_resource: ;
)attn_layer_matmul_readvariableop_resource:@ 8
*attn_layer_biasadd_readvariableop_resource: 
identity��!attn_layer/BiasAdd/ReadVariableOp� attn_layer/MatMul/ReadVariableOp�%features_layer/BiasAdd/ReadVariableOp�$features_layer/MatMul/ReadVariableOp�
$features_layer/MatMul/ReadVariableOpReadVariableOp-features_layer_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
features_layer/MatMulMatMulinputs,features_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
%features_layer/BiasAdd/ReadVariableOpReadVariableOp.features_layer_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
features_layer/BiasAddBiasAddfeatures_layer/MatMul:product:0-features_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
 attn_layer/MatMul/ReadVariableOpReadVariableOp)attn_layer_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
attn_layer/MatMulMatMulinputs(attn_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
!attn_layer/BiasAdd/ReadVariableOpReadVariableOp*attn_layer_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
attn_layer/BiasAddBiasAddattn_layer/MatMul:product:0)attn_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� l
attn_layer/SigmoidSigmoidattn_layer/BiasAdd:output:0*
T0*'
_output_shapes
:��������� u
mulMulfeatures_layer/BiasAdd:output:0attn_layer/Sigmoid:y:0*
T0*'
_output_shapes
:��������� m

SegmentSum
SegmentSummul:z:0inputs_1*
T0*
Tindices0	*'
_output_shapes
:��������� b
IdentityIdentitySegmentSum:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp"^attn_layer/BiasAdd/ReadVariableOp!^attn_layer/MatMul/ReadVariableOp&^features_layer/BiasAdd/ReadVariableOp%^features_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:���������@:���������: : : : 2F
!attn_layer/BiasAdd/ReadVariableOp!attn_layer/BiasAdd/ReadVariableOp2D
 attn_layer/MatMul/ReadVariableOp attn_layer/MatMul/ReadVariableOp2N
%features_layer/BiasAdd/ReadVariableOp%features_layer/BiasAdd/ReadVariableOp2L
$features_layer/MatMul/ReadVariableOp$features_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs
�$
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2989607

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�$
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2988556

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
@__inference_net_layer_call_and_return_conditional_losses_2989385
inputs_0

inputs	
inputs_1
inputs_2	

inputs_2_1	>
0batch_normalization_cast_readvariableop_resource:@
2batch_normalization_cast_1_readvariableop_resource:@
2batch_normalization_cast_2_readvariableop_resource:@
2batch_normalization_cast_3_readvariableop_resource:9
'gcn_conv_matmul_readvariableop_resource:@6
(gcn_conv_biasadd_readvariableop_resource:@@
2batch_normalization_1_cast_readvariableop_resource:@B
4batch_normalization_1_cast_1_readvariableop_resource:@B
4batch_normalization_1_cast_2_readvariableop_resource:@B
4batch_normalization_1_cast_3_readvariableop_resource:@;
)gcn_conv_1_matmul_readvariableop_resource:@@8
*gcn_conv_1_biasadd_readvariableop_resource:@@
2batch_normalization_2_cast_readvariableop_resource:@B
4batch_normalization_2_cast_1_readvariableop_resource:@B
4batch_normalization_2_cast_2_readvariableop_resource:@B
4batch_normalization_2_cast_3_readvariableop_resource:@U
Cglobal_attention_pool_features_layer_matmul_readvariableop_resource:@ R
Dglobal_attention_pool_features_layer_biasadd_readvariableop_resource: Q
?global_attention_pool_attn_layer_matmul_readvariableop_resource:@ N
@global_attention_pool_attn_layer_biasadd_readvariableop_resource: 6
$dense_matmul_readvariableop_resource: @3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:
identity��'batch_normalization/Cast/ReadVariableOp�)batch_normalization/Cast_1/ReadVariableOp�)batch_normalization/Cast_2/ReadVariableOp�)batch_normalization/Cast_3/ReadVariableOp�)batch_normalization_1/Cast/ReadVariableOp�+batch_normalization_1/Cast_1/ReadVariableOp�+batch_normalization_1/Cast_2/ReadVariableOp�+batch_normalization_1/Cast_3/ReadVariableOp�)batch_normalization_2/Cast/ReadVariableOp�+batch_normalization_2/Cast_1/ReadVariableOp�+batch_normalization_2/Cast_2/ReadVariableOp�+batch_normalization_2/Cast_3/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�gcn_conv/BiasAdd/ReadVariableOp�gcn_conv/MatMul/ReadVariableOp�!gcn_conv_1/BiasAdd/ReadVariableOp� gcn_conv_1/MatMul/ReadVariableOp�7global_attention_pool/attn_layer/BiasAdd/ReadVariableOp�6global_attention_pool/attn_layer/MatMul/ReadVariableOp�;global_attention_pool/features_layer/BiasAdd/ReadVariableOp�:global_attention_pool/features_layer/MatMul/ReadVariableOp�
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0�
)batch_normalization/Cast_2/ReadVariableOpReadVariableOp2batch_normalization_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0�
)batch_normalization/Cast_3/ReadVariableOpReadVariableOp2batch_normalization_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
!batch_normalization/batchnorm/addAddV21batch_normalization/Cast_1/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:x
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:�
#batch_normalization/batchnorm/mul_1Mulinputs_0%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
#batch_normalization/batchnorm/mul_2Mul/batch_normalization/Cast/ReadVariableOp:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:�
!batch_normalization/batchnorm/subSub1batch_normalization/Cast_2/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
gcn_conv/MatMul/ReadVariableOpReadVariableOp'gcn_conv_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
gcn_conv/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0&gcn_conv/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
8gcn_conv/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2gcn_conv/MatMul:product:0*
T0*'
_output_shapes
:���������@�
gcn_conv/BiasAdd/ReadVariableOpReadVariableOp(gcn_conv_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
gcn_conv/BiasAddBiasAddBgcn_conv/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0'gcn_conv/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
gcn_conv/ReluRelugcn_conv/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
)batch_normalization_1/Cast/ReadVariableOpReadVariableOp2batch_normalization_1_cast_readvariableop_resource*
_output_shapes
:@*
dtype0�
+batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
+batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes
:@*
dtype0�
+batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes
:@*
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_1/batchnorm/addAddV23batch_normalization_1/Cast_1/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:03batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
%batch_normalization_1/batchnorm/mul_1Mulgcn_conv/Relu:activations:0'batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
%batch_normalization_1/batchnorm/mul_2Mul1batch_normalization_1/Cast/ReadVariableOp:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
#batch_normalization_1/batchnorm/subSub3batch_normalization_1/Cast_2/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
 gcn_conv_1/MatMul/ReadVariableOpReadVariableOp)gcn_conv_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
gcn_conv_1/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:0(gcn_conv_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
:gcn_conv_1/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2gcn_conv_1/MatMul:product:0*
T0*'
_output_shapes
:���������@�
!gcn_conv_1/BiasAdd/ReadVariableOpReadVariableOp*gcn_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
gcn_conv_1/BiasAddBiasAddDgcn_conv_1/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0)gcn_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@f
gcn_conv_1/ReluRelugcn_conv_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
)batch_normalization_2/Cast/ReadVariableOpReadVariableOp2batch_normalization_2_cast_readvariableop_resource*
_output_shapes
:@*
dtype0�
+batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
+batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes
:@*
dtype0�
+batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes
:@*
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_2/batchnorm/addAddV23batch_normalization_2/Cast_1/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:03batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
%batch_normalization_2/batchnorm/mul_1Mulgcn_conv_1/Relu:activations:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
%batch_normalization_2/batchnorm/mul_2Mul1batch_normalization_2/Cast/ReadVariableOp:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
#batch_normalization_2/batchnorm/subSub3batch_normalization_2/Cast_2/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
:global_attention_pool/features_layer/MatMul/ReadVariableOpReadVariableOpCglobal_attention_pool_features_layer_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
+global_attention_pool/features_layer/MatMulMatMul)batch_normalization_2/batchnorm/add_1:z:0Bglobal_attention_pool/features_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
;global_attention_pool/features_layer/BiasAdd/ReadVariableOpReadVariableOpDglobal_attention_pool_features_layer_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
,global_attention_pool/features_layer/BiasAddBiasAdd5global_attention_pool/features_layer/MatMul:product:0Cglobal_attention_pool/features_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
6global_attention_pool/attn_layer/MatMul/ReadVariableOpReadVariableOp?global_attention_pool_attn_layer_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
'global_attention_pool/attn_layer/MatMulMatMul)batch_normalization_2/batchnorm/add_1:z:0>global_attention_pool/attn_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
7global_attention_pool/attn_layer/BiasAdd/ReadVariableOpReadVariableOp@global_attention_pool_attn_layer_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
(global_attention_pool/attn_layer/BiasAddBiasAdd1global_attention_pool/attn_layer/MatMul:product:0?global_attention_pool/attn_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
(global_attention_pool/attn_layer/SigmoidSigmoid1global_attention_pool/attn_layer/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
global_attention_pool/mulMul5global_attention_pool/features_layer/BiasAdd:output:0,global_attention_pool/attn_layer/Sigmoid:y:0*
T0*'
_output_shapes
:��������� �
 global_attention_pool/SegmentSum
SegmentSumglobal_attention_pool/mul:z:0
inputs_2_1*
T0*
Tindices0	*'
_output_shapes
:��������� �
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: @*
dtype0�
dense/MatMulMatMul)global_attention_pool/SegmentSum:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp*^batch_normalization/Cast_2/ReadVariableOp*^batch_normalization/Cast_3/ReadVariableOp*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp,^batch_normalization_1/Cast_2/ReadVariableOp,^batch_normalization_1/Cast_3/ReadVariableOp*^batch_normalization_2/Cast/ReadVariableOp,^batch_normalization_2/Cast_1/ReadVariableOp,^batch_normalization_2/Cast_2/ReadVariableOp,^batch_normalization_2/Cast_3/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp ^gcn_conv/BiasAdd/ReadVariableOp^gcn_conv/MatMul/ReadVariableOp"^gcn_conv_1/BiasAdd/ReadVariableOp!^gcn_conv_1/MatMul/ReadVariableOp8^global_attention_pool/attn_layer/BiasAdd/ReadVariableOp7^global_attention_pool/attn_layer/MatMul/ReadVariableOp<^global_attention_pool/features_layer/BiasAdd/ReadVariableOp;^global_attention_pool/features_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes|
z:���������:���������:���������::���������: : : : : : : : : : : : : : : : : : : : : : : : 2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2V
)batch_normalization/Cast_2/ReadVariableOp)batch_normalization/Cast_2/ReadVariableOp2V
)batch_normalization/Cast_3/ReadVariableOp)batch_normalization/Cast_3/ReadVariableOp2V
)batch_normalization_1/Cast/ReadVariableOp)batch_normalization_1/Cast/ReadVariableOp2Z
+batch_normalization_1/Cast_1/ReadVariableOp+batch_normalization_1/Cast_1/ReadVariableOp2Z
+batch_normalization_1/Cast_2/ReadVariableOp+batch_normalization_1/Cast_2/ReadVariableOp2Z
+batch_normalization_1/Cast_3/ReadVariableOp+batch_normalization_1/Cast_3/ReadVariableOp2V
)batch_normalization_2/Cast/ReadVariableOp)batch_normalization_2/Cast/ReadVariableOp2Z
+batch_normalization_2/Cast_1/ReadVariableOp+batch_normalization_2/Cast_1/ReadVariableOp2Z
+batch_normalization_2/Cast_2/ReadVariableOp+batch_normalization_2/Cast_2/ReadVariableOp2Z
+batch_normalization_2/Cast_3/ReadVariableOp+batch_normalization_2/Cast_3/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2B
gcn_conv/BiasAdd/ReadVariableOpgcn_conv/BiasAdd/ReadVariableOp2@
gcn_conv/MatMul/ReadVariableOpgcn_conv/MatMul/ReadVariableOp2F
!gcn_conv_1/BiasAdd/ReadVariableOp!gcn_conv_1/BiasAdd/ReadVariableOp2D
 gcn_conv_1/MatMul/ReadVariableOp gcn_conv_1/MatMul/ReadVariableOp2r
7global_attention_pool/attn_layer/BiasAdd/ReadVariableOp7global_attention_pool/attn_layer/BiasAdd/ReadVariableOp2p
6global_attention_pool/attn_layer/MatMul/ReadVariableOp6global_attention_pool/attn_layer/MatMul/ReadVariableOp2z
;global_attention_pool/features_layer/BiasAdd/ReadVariableOp;global_attention_pool/features_layer/BiasAdd/ReadVariableOp2x
:global_attention_pool/features_layer/MatMul/ReadVariableOp:global_attention_pool/features_layer/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs:MI
#
_output_shapes
:���������
"
_user_specified_name
inputs_2
�$
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2989714

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@*
cast_readvariableop_resource:@,
cast_1_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
G__inference_gcn_conv_1_layer_call_and_return_conditional_losses_2988800

inputs
inputs_1	
inputs_2
inputs_3	0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1inputs_2inputs_3MatMul:product:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAdd9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������@:���������:���������:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�
�
%__inference_net_layer_call_fn_2989228
inputs_0

inputs	
inputs_1
inputs_2	

inputs_2_1	
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@ 

unknown_16: 

unknown_17:@ 

unknown_18: 

unknown_19: @

unknown_20:@

unknown_21:@

unknown_22:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2
inputs_2_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*(
Tin!
2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_net_layer_call_and_return_conditional_losses_2988879o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes|
z:���������:���������:���������::���������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs:MI
#
_output_shapes
:���������
"
_user_specified_name
inputs_2
�

�
,__inference_gcn_conv_1_layer_call_fn_2989726
inputs_0

inputs	
inputs_1
inputs_2	
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2unknown	unknown_0*
Tin

2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_gcn_conv_1_layer_call_and_return_conditional_losses_2988800o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������@:���������:���������:: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������@
"
_user_specified_name
inputs_0:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�
�
)__inference_dense_1_layer_call_fn_2989884

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_2988872o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
E__inference_gcn_conv_layer_call_and_return_conditional_losses_2988770

inputs
inputs_1	
inputs_2
inputs_3	0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1inputs_2inputs_3MatMul:product:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAdd9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_2989171

args_0
args_0_1	
args_0_2
args_0_3	
args_0_4	
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@ 

unknown_16: 

unknown_17:@ 

unknown_18: 

unknown_19: @

unknown_20:@

unknown_21:@

unknown_22:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallargs_0args_0_1args_0_2args_0_3args_0_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*(
Tin!
2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_2988485o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes|
z:���������:���������:���������::���������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameargs_0:QM
'
_output_shapes
:���������
"
_user_specified_name
args_0_1:MI
#
_output_shapes
:���������
"
_user_specified_name
args_0_2:D@

_output_shapes
:
"
_user_specified_name
args_0_3:MI
#
_output_shapes
:���������
"
_user_specified_name
args_0_4
�
�
5__inference_batch_normalization_layer_call_fn_2989553

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2988556o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�7
�
@__inference_net_layer_call_and_return_conditional_losses_2989061

inputs
inputs_1	
inputs_2
inputs_3	
inputs_4	)
batch_normalization_2989004:)
batch_normalization_2989006:)
batch_normalization_2989008:)
batch_normalization_2989010:"
gcn_conv_2989013:@
gcn_conv_2989015:@+
batch_normalization_1_2989018:@+
batch_normalization_1_2989020:@+
batch_normalization_1_2989022:@+
batch_normalization_1_2989024:@$
gcn_conv_1_2989027:@@ 
gcn_conv_1_2989029:@+
batch_normalization_2_2989032:@+
batch_normalization_2_2989034:@+
batch_normalization_2_2989036:@+
batch_normalization_2_2989038:@/
global_attention_pool_2989041:@ +
global_attention_pool_2989043: /
global_attention_pool_2989045:@ +
global_attention_pool_2989047: 
dense_2989050: @
dense_2989052:@!
dense_1_2989055:@
dense_1_2989057:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall� gcn_conv/StatefulPartitionedCall�"gcn_conv_1/StatefulPartitionedCall�-global_attention_pool/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_2989004batch_normalization_2989006batch_normalization_2989008batch_normalization_2989010*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2988556�
 gcn_conv/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0inputs_1inputs_2inputs_3gcn_conv_2989013gcn_conv_2989015*
Tin

2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_gcn_conv_layer_call_and_return_conditional_losses_2988770�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)gcn_conv/StatefulPartitionedCall:output:0batch_normalization_1_2989018batch_normalization_1_2989020batch_normalization_1_2989022batch_normalization_1_2989024*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2988638�
"gcn_conv_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0inputs_1inputs_2inputs_3gcn_conv_1_2989027gcn_conv_1_2989029*
Tin

2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_gcn_conv_1_layer_call_and_return_conditional_losses_2988800�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall+gcn_conv_1/StatefulPartitionedCall:output:0batch_normalization_2_2989032batch_normalization_2_2989034batch_normalization_2_2989036batch_normalization_2_2989038*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2988720�
-global_attention_pool/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0inputs_4global_attention_pool_2989041global_attention_pool_2989043global_attention_pool_2989045global_attention_pool_2989047*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_global_attention_pool_layer_call_and_return_conditional_losses_2988835�
dense/StatefulPartitionedCallStatefulPartitionedCall6global_attention_pool/StatefulPartitionedCall:output:0dense_2989050dense_2989052*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_2988856�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_2989055dense_1_2989057*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_2988872w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall!^gcn_conv/StatefulPartitionedCall#^gcn_conv_1/StatefulPartitionedCall.^global_attention_pool/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes|
z:���������:���������:���������::���������: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 gcn_conv/StatefulPartitionedCall gcn_conv/StatefulPartitionedCall2H
"gcn_conv_1/StatefulPartitionedCall"gcn_conv_1/StatefulPartitionedCall2^
-global_attention_pool/StatefulPartitionedCall-global_attention_pool/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
"__inference__wrapped_model_2988485

args_0
args_0_1	
args_0_2
args_0_3	
args_0_4	B
4net_batch_normalization_cast_readvariableop_resource:D
6net_batch_normalization_cast_1_readvariableop_resource:D
6net_batch_normalization_cast_2_readvariableop_resource:D
6net_batch_normalization_cast_3_readvariableop_resource:=
+net_gcn_conv_matmul_readvariableop_resource:@:
,net_gcn_conv_biasadd_readvariableop_resource:@D
6net_batch_normalization_1_cast_readvariableop_resource:@F
8net_batch_normalization_1_cast_1_readvariableop_resource:@F
8net_batch_normalization_1_cast_2_readvariableop_resource:@F
8net_batch_normalization_1_cast_3_readvariableop_resource:@?
-net_gcn_conv_1_matmul_readvariableop_resource:@@<
.net_gcn_conv_1_biasadd_readvariableop_resource:@D
6net_batch_normalization_2_cast_readvariableop_resource:@F
8net_batch_normalization_2_cast_1_readvariableop_resource:@F
8net_batch_normalization_2_cast_2_readvariableop_resource:@F
8net_batch_normalization_2_cast_3_readvariableop_resource:@Y
Gnet_global_attention_pool_features_layer_matmul_readvariableop_resource:@ V
Hnet_global_attention_pool_features_layer_biasadd_readvariableop_resource: U
Cnet_global_attention_pool_attn_layer_matmul_readvariableop_resource:@ R
Dnet_global_attention_pool_attn_layer_biasadd_readvariableop_resource: :
(net_dense_matmul_readvariableop_resource: @7
)net_dense_biasadd_readvariableop_resource:@<
*net_dense_1_matmul_readvariableop_resource:@9
+net_dense_1_biasadd_readvariableop_resource:
identity��+net/batch_normalization/Cast/ReadVariableOp�-net/batch_normalization/Cast_1/ReadVariableOp�-net/batch_normalization/Cast_2/ReadVariableOp�-net/batch_normalization/Cast_3/ReadVariableOp�-net/batch_normalization_1/Cast/ReadVariableOp�/net/batch_normalization_1/Cast_1/ReadVariableOp�/net/batch_normalization_1/Cast_2/ReadVariableOp�/net/batch_normalization_1/Cast_3/ReadVariableOp�-net/batch_normalization_2/Cast/ReadVariableOp�/net/batch_normalization_2/Cast_1/ReadVariableOp�/net/batch_normalization_2/Cast_2/ReadVariableOp�/net/batch_normalization_2/Cast_3/ReadVariableOp� net/dense/BiasAdd/ReadVariableOp�net/dense/MatMul/ReadVariableOp�"net/dense_1/BiasAdd/ReadVariableOp�!net/dense_1/MatMul/ReadVariableOp�#net/gcn_conv/BiasAdd/ReadVariableOp�"net/gcn_conv/MatMul/ReadVariableOp�%net/gcn_conv_1/BiasAdd/ReadVariableOp�$net/gcn_conv_1/MatMul/ReadVariableOp�;net/global_attention_pool/attn_layer/BiasAdd/ReadVariableOp�:net/global_attention_pool/attn_layer/MatMul/ReadVariableOp�?net/global_attention_pool/features_layer/BiasAdd/ReadVariableOp�>net/global_attention_pool/features_layer/MatMul/ReadVariableOp�
+net/batch_normalization/Cast/ReadVariableOpReadVariableOp4net_batch_normalization_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
-net/batch_normalization/Cast_1/ReadVariableOpReadVariableOp6net_batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0�
-net/batch_normalization/Cast_2/ReadVariableOpReadVariableOp6net_batch_normalization_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0�
-net/batch_normalization/Cast_3/ReadVariableOpReadVariableOp6net_batch_normalization_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0l
'net/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
%net/batch_normalization/batchnorm/addAddV25net/batch_normalization/Cast_1/ReadVariableOp:value:00net/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
'net/batch_normalization/batchnorm/RsqrtRsqrt)net/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:�
%net/batch_normalization/batchnorm/mulMul+net/batch_normalization/batchnorm/Rsqrt:y:05net/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:�
'net/batch_normalization/batchnorm/mul_1Mulargs_0)net/batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
'net/batch_normalization/batchnorm/mul_2Mul3net/batch_normalization/Cast/ReadVariableOp:value:0)net/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:�
%net/batch_normalization/batchnorm/subSub5net/batch_normalization/Cast_2/ReadVariableOp:value:0+net/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
'net/batch_normalization/batchnorm/add_1AddV2+net/batch_normalization/batchnorm/mul_1:z:0)net/batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
"net/gcn_conv/MatMul/ReadVariableOpReadVariableOp+net_gcn_conv_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
net/gcn_conv/MatMulMatMul+net/batch_normalization/batchnorm/add_1:z:0*net/gcn_conv/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
<net/gcn_conv/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulargs_0_1args_0_2args_0_3net/gcn_conv/MatMul:product:0*
T0*'
_output_shapes
:���������@�
#net/gcn_conv/BiasAdd/ReadVariableOpReadVariableOp,net_gcn_conv_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
net/gcn_conv/BiasAddBiasAddFnet/gcn_conv/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0+net/gcn_conv/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@j
net/gcn_conv/ReluRelunet/gcn_conv/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
-net/batch_normalization_1/Cast/ReadVariableOpReadVariableOp6net_batch_normalization_1_cast_readvariableop_resource*
_output_shapes
:@*
dtype0�
/net/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp8net_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
/net/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp8net_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes
:@*
dtype0�
/net/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp8net_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes
:@*
dtype0n
)net/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
'net/batch_normalization_1/batchnorm/addAddV27net/batch_normalization_1/Cast_1/ReadVariableOp:value:02net/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@�
)net/batch_normalization_1/batchnorm/RsqrtRsqrt+net/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@�
'net/batch_normalization_1/batchnorm/mulMul-net/batch_normalization_1/batchnorm/Rsqrt:y:07net/batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
)net/batch_normalization_1/batchnorm/mul_1Mulnet/gcn_conv/Relu:activations:0+net/batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
)net/batch_normalization_1/batchnorm/mul_2Mul5net/batch_normalization_1/Cast/ReadVariableOp:value:0+net/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
'net/batch_normalization_1/batchnorm/subSub7net/batch_normalization_1/Cast_2/ReadVariableOp:value:0-net/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
)net/batch_normalization_1/batchnorm/add_1AddV2-net/batch_normalization_1/batchnorm/mul_1:z:0+net/batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
$net/gcn_conv_1/MatMul/ReadVariableOpReadVariableOp-net_gcn_conv_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
net/gcn_conv_1/MatMulMatMul-net/batch_normalization_1/batchnorm/add_1:z:0,net/gcn_conv_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
>net/gcn_conv_1/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulargs_0_1args_0_2args_0_3net/gcn_conv_1/MatMul:product:0*
T0*'
_output_shapes
:���������@�
%net/gcn_conv_1/BiasAdd/ReadVariableOpReadVariableOp.net_gcn_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
net/gcn_conv_1/BiasAddBiasAddHnet/gcn_conv_1/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0-net/gcn_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@n
net/gcn_conv_1/ReluRelunet/gcn_conv_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
-net/batch_normalization_2/Cast/ReadVariableOpReadVariableOp6net_batch_normalization_2_cast_readvariableop_resource*
_output_shapes
:@*
dtype0�
/net/batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp8net_batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
/net/batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp8net_batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes
:@*
dtype0�
/net/batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp8net_batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes
:@*
dtype0n
)net/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
'net/batch_normalization_2/batchnorm/addAddV27net/batch_normalization_2/Cast_1/ReadVariableOp:value:02net/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@�
)net/batch_normalization_2/batchnorm/RsqrtRsqrt+net/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@�
'net/batch_normalization_2/batchnorm/mulMul-net/batch_normalization_2/batchnorm/Rsqrt:y:07net/batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
)net/batch_normalization_2/batchnorm/mul_1Mul!net/gcn_conv_1/Relu:activations:0+net/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
)net/batch_normalization_2/batchnorm/mul_2Mul5net/batch_normalization_2/Cast/ReadVariableOp:value:0+net/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
'net/batch_normalization_2/batchnorm/subSub7net/batch_normalization_2/Cast_2/ReadVariableOp:value:0-net/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
)net/batch_normalization_2/batchnorm/add_1AddV2-net/batch_normalization_2/batchnorm/mul_1:z:0+net/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
>net/global_attention_pool/features_layer/MatMul/ReadVariableOpReadVariableOpGnet_global_attention_pool_features_layer_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
/net/global_attention_pool/features_layer/MatMulMatMul-net/batch_normalization_2/batchnorm/add_1:z:0Fnet/global_attention_pool/features_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
?net/global_attention_pool/features_layer/BiasAdd/ReadVariableOpReadVariableOpHnet_global_attention_pool_features_layer_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
0net/global_attention_pool/features_layer/BiasAddBiasAdd9net/global_attention_pool/features_layer/MatMul:product:0Gnet/global_attention_pool/features_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
:net/global_attention_pool/attn_layer/MatMul/ReadVariableOpReadVariableOpCnet_global_attention_pool_attn_layer_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
+net/global_attention_pool/attn_layer/MatMulMatMul-net/batch_normalization_2/batchnorm/add_1:z:0Bnet/global_attention_pool/attn_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
;net/global_attention_pool/attn_layer/BiasAdd/ReadVariableOpReadVariableOpDnet_global_attention_pool_attn_layer_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
,net/global_attention_pool/attn_layer/BiasAddBiasAdd5net/global_attention_pool/attn_layer/MatMul:product:0Cnet/global_attention_pool/attn_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
,net/global_attention_pool/attn_layer/SigmoidSigmoid5net/global_attention_pool/attn_layer/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
net/global_attention_pool/mulMul9net/global_attention_pool/features_layer/BiasAdd:output:00net/global_attention_pool/attn_layer/Sigmoid:y:0*
T0*'
_output_shapes
:��������� �
$net/global_attention_pool/SegmentSum
SegmentSum!net/global_attention_pool/mul:z:0args_0_4*
T0*
Tindices0	*'
_output_shapes
:��������� �
net/dense/MatMul/ReadVariableOpReadVariableOp(net_dense_matmul_readvariableop_resource*
_output_shapes

: @*
dtype0�
net/dense/MatMulMatMul-net/global_attention_pool/SegmentSum:output:0'net/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
 net/dense/BiasAdd/ReadVariableOpReadVariableOp)net_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
net/dense/BiasAddBiasAddnet/dense/MatMul:product:0(net/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@d
net/dense/ReluRelunet/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
!net/dense_1/MatMul/ReadVariableOpReadVariableOp*net_dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
net/dense_1/MatMulMatMulnet/dense/Relu:activations:0)net/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"net/dense_1/BiasAdd/ReadVariableOpReadVariableOp+net_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
net/dense_1/BiasAddBiasAddnet/dense_1/MatMul:product:0*net/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������k
IdentityIdentitynet/dense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������	
NoOpNoOp,^net/batch_normalization/Cast/ReadVariableOp.^net/batch_normalization/Cast_1/ReadVariableOp.^net/batch_normalization/Cast_2/ReadVariableOp.^net/batch_normalization/Cast_3/ReadVariableOp.^net/batch_normalization_1/Cast/ReadVariableOp0^net/batch_normalization_1/Cast_1/ReadVariableOp0^net/batch_normalization_1/Cast_2/ReadVariableOp0^net/batch_normalization_1/Cast_3/ReadVariableOp.^net/batch_normalization_2/Cast/ReadVariableOp0^net/batch_normalization_2/Cast_1/ReadVariableOp0^net/batch_normalization_2/Cast_2/ReadVariableOp0^net/batch_normalization_2/Cast_3/ReadVariableOp!^net/dense/BiasAdd/ReadVariableOp ^net/dense/MatMul/ReadVariableOp#^net/dense_1/BiasAdd/ReadVariableOp"^net/dense_1/MatMul/ReadVariableOp$^net/gcn_conv/BiasAdd/ReadVariableOp#^net/gcn_conv/MatMul/ReadVariableOp&^net/gcn_conv_1/BiasAdd/ReadVariableOp%^net/gcn_conv_1/MatMul/ReadVariableOp<^net/global_attention_pool/attn_layer/BiasAdd/ReadVariableOp;^net/global_attention_pool/attn_layer/MatMul/ReadVariableOp@^net/global_attention_pool/features_layer/BiasAdd/ReadVariableOp?^net/global_attention_pool/features_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes|
z:���������:���������:���������::���������: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+net/batch_normalization/Cast/ReadVariableOp+net/batch_normalization/Cast/ReadVariableOp2^
-net/batch_normalization/Cast_1/ReadVariableOp-net/batch_normalization/Cast_1/ReadVariableOp2^
-net/batch_normalization/Cast_2/ReadVariableOp-net/batch_normalization/Cast_2/ReadVariableOp2^
-net/batch_normalization/Cast_3/ReadVariableOp-net/batch_normalization/Cast_3/ReadVariableOp2^
-net/batch_normalization_1/Cast/ReadVariableOp-net/batch_normalization_1/Cast/ReadVariableOp2b
/net/batch_normalization_1/Cast_1/ReadVariableOp/net/batch_normalization_1/Cast_1/ReadVariableOp2b
/net/batch_normalization_1/Cast_2/ReadVariableOp/net/batch_normalization_1/Cast_2/ReadVariableOp2b
/net/batch_normalization_1/Cast_3/ReadVariableOp/net/batch_normalization_1/Cast_3/ReadVariableOp2^
-net/batch_normalization_2/Cast/ReadVariableOp-net/batch_normalization_2/Cast/ReadVariableOp2b
/net/batch_normalization_2/Cast_1/ReadVariableOp/net/batch_normalization_2/Cast_1/ReadVariableOp2b
/net/batch_normalization_2/Cast_2/ReadVariableOp/net/batch_normalization_2/Cast_2/ReadVariableOp2b
/net/batch_normalization_2/Cast_3/ReadVariableOp/net/batch_normalization_2/Cast_3/ReadVariableOp2D
 net/dense/BiasAdd/ReadVariableOp net/dense/BiasAdd/ReadVariableOp2B
net/dense/MatMul/ReadVariableOpnet/dense/MatMul/ReadVariableOp2H
"net/dense_1/BiasAdd/ReadVariableOp"net/dense_1/BiasAdd/ReadVariableOp2F
!net/dense_1/MatMul/ReadVariableOp!net/dense_1/MatMul/ReadVariableOp2J
#net/gcn_conv/BiasAdd/ReadVariableOp#net/gcn_conv/BiasAdd/ReadVariableOp2H
"net/gcn_conv/MatMul/ReadVariableOp"net/gcn_conv/MatMul/ReadVariableOp2N
%net/gcn_conv_1/BiasAdd/ReadVariableOp%net/gcn_conv_1/BiasAdd/ReadVariableOp2L
$net/gcn_conv_1/MatMul/ReadVariableOp$net/gcn_conv_1/MatMul/ReadVariableOp2z
;net/global_attention_pool/attn_layer/BiasAdd/ReadVariableOp;net/global_attention_pool/attn_layer/BiasAdd/ReadVariableOp2x
:net/global_attention_pool/attn_layer/MatMul/ReadVariableOp:net/global_attention_pool/attn_layer/MatMul/ReadVariableOp2�
?net/global_attention_pool/features_layer/BiasAdd/ReadVariableOp?net/global_attention_pool/features_layer/BiasAdd/ReadVariableOp2�
>net/global_attention_pool/features_layer/MatMul/ReadVariableOp>net/global_attention_pool/features_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameargs_0:OK
'
_output_shapes
:���������
 
_user_specified_nameargs_0:KG
#
_output_shapes
:���������
 
_user_specified_nameargs_0:B>

_output_shapes
:
 
_user_specified_nameargs_0:KG
#
_output_shapes
:���������
 
_user_specified_nameargs_0
�$
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2988720

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@*
cast_readvariableop_resource:@,
cast_1_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2989573

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
9
args_0/
serving_default_args_0:0���������
=
args_0_11
serving_default_args_0_1:0	���������
9
args_0_2-
serving_default_args_0_2:0���������
0
args_0_3$
serving_default_args_0_3:0	
9
args_0_4-
serving_default_args_0_4:0	���������<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

batchnorm1
		conv1


batchnorm2
	conv2

batchnorm3
global_pool

dense1

dense2

signatures"
_tf_keras_model
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
 15
!16
"17
#18
$19
%20
&21
'22
(23"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
8
9
!10
"11
#12
$13
%14
&15
'16
(17"
trackable_list_wrapper
 "
trackable_list_wrapper
�
)non_trainable_variables

*layers
+metrics
,layer_regularization_losses
-layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
.trace_0
/trace_12�
%__inference_net_layer_call_fn_2989228
%__inference_net_layer_call_fn_2989285�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z.trace_0z/trace_1
�
0trace_0
1trace_12�
@__inference_net_layer_call_and_return_conditional_losses_2989385
@__inference_net_layer_call_and_return_conditional_losses_2989527�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z0trace_0z1trace_1
�B�
"__inference__wrapped_model_2988485args_0args_0_1args_0_2args_0_3args_0_4"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses
8axis
	gamma
beta
moving_mean
moving_variance"
_tf_keras_layer
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?kwargs_keys

kernel
bias"
_tf_keras_layer
�
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses
Faxis
	gamma
beta
moving_mean
moving_variance"
_tf_keras_layer
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses
Mkwargs_keys

kernel
bias"
_tf_keras_layer
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
Taxis
	gamma
beta
moving_mean
 moving_variance"
_tf_keras_layer
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses
[features_layer
\attention_layer"
_tf_keras_layer
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses

%kernel
&bias"
_tf_keras_layer
�
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

'kernel
(bias"
_tf_keras_layer
,
iserving_default"
signature_map
+:)2net/batch_normalization/gamma
*:(2net/batch_normalization/beta
3:1 (2#net/batch_normalization/moving_mean
7:5 (2'net/batch_normalization/moving_variance
%:#@2net/gcn_conv/kernel
:@2net/gcn_conv/bias
-:+@2net/batch_normalization_1/gamma
,:*@2net/batch_normalization_1/beta
5:3@ (2%net/batch_normalization_1/moving_mean
9:7@ (2)net/batch_normalization_1/moving_variance
':%@@2net/gcn_conv_1/kernel
!:@2net/gcn_conv_1/bias
-:+@2net/batch_normalization_2/gamma
,:*@2net/batch_normalization_2/beta
5:3@ (2%net/batch_normalization_2/moving_mean
9:7@ (2)net/batch_normalization_2/moving_variance
A:?@ 2/net/global_attention_pool/features_layer/kernel
;:9 2-net/global_attention_pool/features_layer/bias
=:;@ 2+net/global_attention_pool/attn_layer/kernel
7:5 2)net/global_attention_pool/attn_layer/bias
":  @2net/dense/kernel
:@2net/dense/bias
$:"@2net/dense_1/kernel
:2net/dense_1/bias
J
0
1
2
3
4
 5"
trackable_list_wrapper
X
0
	1

2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_net_layer_call_fn_2989228inputs_0inputsinputs_1inputs_2
inputs_2_1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
%__inference_net_layer_call_fn_2989285inputs_0inputsinputs_1inputs_2
inputs_2_1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
@__inference_net_layer_call_and_return_conditional_losses_2989385inputs_0inputsinputs_1inputs_2
inputs_2_1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
@__inference_net_layer_call_and_return_conditional_losses_2989527inputs_0inputsinputs_1inputs_2
inputs_2_1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
�
otrace_0
ptrace_12�
5__inference_batch_normalization_layer_call_fn_2989540
5__inference_batch_normalization_layer_call_fn_2989553�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zotrace_0zptrace_1
�
qtrace_0
rtrace_12�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2989573
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2989607�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zqtrace_0zrtrace_1
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�
xtrace_02�
*__inference_gcn_conv_layer_call_fn_2989619�
���
FullArgSpec
args�

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zxtrace_0
�
ytrace_02�
E__inference_gcn_conv_layer_call_and_return_conditional_losses_2989634�
���
FullArgSpec
args�

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zytrace_0
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
�
trace_0
�trace_12�
7__inference_batch_normalization_1_layer_call_fn_2989647
7__inference_batch_normalization_1_layer_call_fn_2989660�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2989680
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2989714�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_gcn_conv_1_layer_call_fn_2989726�
���
FullArgSpec
args�

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_gcn_conv_1_layer_call_and_return_conditional_losses_2989741�
���
FullArgSpec
args�

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
<
0
1
2
 3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_batch_normalization_2_layer_call_fn_2989754
7__inference_batch_normalization_2_layer_call_fn_2989767�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2989787
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2989821�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
<
!0
"1
#2
$3"
trackable_list_wrapper
<
!0
"1
#2
$3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
7__inference_global_attention_pool_layer_call_fn_2989835�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
R__inference_global_attention_pool_layer_call_and_return_conditional_losses_2989855�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

!kernel
"bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

#kernel
$bias"
_tf_keras_layer
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_dense_layer_call_fn_2989864�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_dense_layer_call_and_return_conditional_losses_2989875�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_1_layer_call_fn_2989884�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_1_layer_call_and_return_conditional_losses_2989894�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�B�
%__inference_signature_wrapper_2989171args_0args_0_1args_0_2args_0_3args_0_4"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
5__inference_batch_normalization_layer_call_fn_2989540inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
5__inference_batch_normalization_layer_call_fn_2989553inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2989573inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2989607inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_gcn_conv_layer_call_fn_2989619inputs_0inputsinputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_gcn_conv_layer_call_and_return_conditional_losses_2989634inputs_0inputsinputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
7__inference_batch_normalization_1_layer_call_fn_2989647inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
7__inference_batch_normalization_1_layer_call_fn_2989660inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2989680inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2989714inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_gcn_conv_1_layer_call_fn_2989726inputs_0inputsinputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_gcn_conv_1_layer_call_and_return_conditional_losses_2989741inputs_0inputsinputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
7__inference_batch_normalization_2_layer_call_fn_2989754inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
7__inference_batch_normalization_2_layer_call_fn_2989767inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2989787inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2989821inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
7__inference_global_attention_pool_layer_call_fn_2989835inputs_0inputs_1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_global_attention_pool_layer_call_and_return_conditional_losses_2989855inputs_0inputs_1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_dense_layer_call_fn_2989864inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dense_layer_call_and_return_conditional_losses_2989875inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_1_layer_call_fn_2989884inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_1_layer_call_and_return_conditional_losses_2989894inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper�
"__inference__wrapped_model_2988485� !"#$%&'(���
���
���
"�
args_0_0���������
B�?'�$
�������������������
�SparseTensorSpec 
�
args_0_2���������	
� "3�0
.
output_1"�
output_1����������
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2989680i3�0
)�&
 �
inputs���������@
p 
� ",�)
"�
tensor_0���������@
� �
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2989714i3�0
)�&
 �
inputs���������@
p
� ",�)
"�
tensor_0���������@
� �
7__inference_batch_normalization_1_layer_call_fn_2989647^3�0
)�&
 �
inputs���������@
p 
� "!�
unknown���������@�
7__inference_batch_normalization_1_layer_call_fn_2989660^3�0
)�&
 �
inputs���������@
p
� "!�
unknown���������@�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2989787i 3�0
)�&
 �
inputs���������@
p 
� ",�)
"�
tensor_0���������@
� �
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2989821i 3�0
)�&
 �
inputs���������@
p
� ",�)
"�
tensor_0���������@
� �
7__inference_batch_normalization_2_layer_call_fn_2989754^ 3�0
)�&
 �
inputs���������@
p 
� "!�
unknown���������@�
7__inference_batch_normalization_2_layer_call_fn_2989767^ 3�0
)�&
 �
inputs���������@
p
� "!�
unknown���������@�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2989573i3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2989607i3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
5__inference_batch_normalization_layer_call_fn_2989540^3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
5__inference_batch_normalization_layer_call_fn_2989553^3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
D__inference_dense_1_layer_call_and_return_conditional_losses_2989894c'(/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������
� �
)__inference_dense_1_layer_call_fn_2989884X'(/�,
%�"
 �
inputs���������@
� "!�
unknown����������
B__inference_dense_layer_call_and_return_conditional_losses_2989875c%&/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0���������@
� �
'__inference_dense_layer_call_fn_2989864X%&/�,
%�"
 �
inputs��������� 
� "!�
unknown���������@�
G__inference_gcn_conv_1_layer_call_and_return_conditional_losses_2989741�z�w
p�m
k�h
"�
inputs_0���������@
B�?'�$
�������������������
�SparseTensorSpec 
� ",�)
"�
tensor_0���������@
� �
,__inference_gcn_conv_1_layer_call_fn_2989726�z�w
p�m
k�h
"�
inputs_0���������@
B�?'�$
�������������������
�SparseTensorSpec 
� "!�
unknown���������@�
E__inference_gcn_conv_layer_call_and_return_conditional_losses_2989634�z�w
p�m
k�h
"�
inputs_0���������
B�?'�$
�������������������
�SparseTensorSpec 
� ",�)
"�
tensor_0���������@
� �
*__inference_gcn_conv_layer_call_fn_2989619�z�w
p�m
k�h
"�
inputs_0���������
B�?'�$
�������������������
�SparseTensorSpec 
� "!�
unknown���������@�
R__inference_global_attention_pool_layer_call_and_return_conditional_losses_2989855�!"#$V�S
L�I
G�D
"�
inputs_0���������@
�
inputs_1���������	
� ",�)
"�
tensor_0��������� 
� �
7__inference_global_attention_pool_layer_call_fn_2989835�!"#$V�S
L�I
G�D
"�
inputs_0���������@
�
inputs_1���������	
� "!�
unknown��������� �
@__inference_net_layer_call_and_return_conditional_losses_2989385� !"#$%&'(���
���
���
"�
inputs_0���������
B�?'�$
�������������������
�SparseTensorSpec 
�
inputs_2���������	
�

trainingp ",�)
"�
tensor_0���������
� �
@__inference_net_layer_call_and_return_conditional_losses_2989527� !"#$%&'(���
���
���
"�
inputs_0���������
B�?'�$
�������������������
�SparseTensorSpec 
�
inputs_2���������	
�

trainingp",�)
"�
tensor_0���������
� �
%__inference_net_layer_call_fn_2989228� !"#$%&'(���
���
���
"�
inputs_0���������
B�?'�$
�������������������
�SparseTensorSpec 
�
inputs_2���������	
�

trainingp "!�
unknown����������
%__inference_net_layer_call_fn_2989285� !"#$%&'(���
���
���
"�
inputs_0���������
B�?'�$
�������������������
�SparseTensorSpec 
�
inputs_2���������	
�

trainingp"!�
unknown����������
%__inference_signature_wrapper_2989171� !"#$%&'(���
� 
���
*
args_0 �
args_0���������
.
args_0_1"�
args_0_1���������	
*
args_0_2�
args_0_2���������
!
args_0_3�
args_0_3	
*
args_0_4�
args_0_4���������	"3�0
.
output_1"�
output_1���������