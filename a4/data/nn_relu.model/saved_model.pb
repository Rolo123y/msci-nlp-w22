É
¿
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(
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
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
¥
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68½

word_embedding_layer/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¸×d*0
shared_name!word_embedding_layer/embeddings

3word_embedding_layer/embeddings/Read/ReadVariableOpReadVariableOpword_embedding_layer/embeddings* 
_output_shapes
:
¸×d*
dtype0
~
relu_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*"
shared_namerelu_layer/kernel
w
%relu_layer/kernel/Read/ReadVariableOpReadVariableOprelu_layer/kernel*
_output_shapes

:d*
dtype0
v
relu_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namerelu_layer/bias
o
#relu_layer/bias/Read/ReadVariableOpReadVariableOprelu_layer/bias*
_output_shapes
:*
dtype0

output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameoutput_layer/kernel
{
'output_layer/kernel/Read/ReadVariableOpReadVariableOpoutput_layer/kernel*
_output_shapes

:*
dtype0
z
output_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameoutput_layer/bias
s
%output_layer/bias/Read/ReadVariableOpReadVariableOpoutput_layer/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/relu_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*)
shared_nameAdam/relu_layer/kernel/m

,Adam/relu_layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/relu_layer/kernel/m*
_output_shapes

:d*
dtype0

Adam/relu_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/relu_layer/bias/m
}
*Adam/relu_layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/relu_layer/bias/m*
_output_shapes
:*
dtype0

Adam/output_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameAdam/output_layer/kernel/m

.Adam/output_layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output_layer/kernel/m*
_output_shapes

:*
dtype0

Adam/output_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/output_layer/bias/m

,Adam/output_layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/output_layer/bias/m*
_output_shapes
:*
dtype0

Adam/relu_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*)
shared_nameAdam/relu_layer/kernel/v

,Adam/relu_layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/relu_layer/kernel/v*
_output_shapes

:d*
dtype0

Adam/relu_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/relu_layer/bias/v
}
*Adam/relu_layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/relu_layer/bias/v*
_output_shapes
:*
dtype0

Adam/output_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameAdam/output_layer/kernel/v

.Adam/output_layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output_layer/kernel/v*
_output_shapes

:*
dtype0

Adam/output_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/output_layer/bias/v

,Adam/output_layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/output_layer/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
0
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ð/
valueÆ/BÃ/ B¼/
Û
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
 

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
¦

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses*
¥
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(_random_generator
)__call__
**&call_and_return_all_conditional_losses* 
¦

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses*

3iter

4beta_1

5beta_2
	6decay
7learning_ratemcmd+me,mfvgvh+vi,vj*
'
0
1
2
+3
,4*
 
0
1
+2
,3*
	
80* 
°
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

>serving_default* 
sm
VARIABLE_VALUEword_embedding_layer/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

0*
* 
* 

?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUErelu_layer/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUErelu_layer/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
	
80* 

Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
$	variables
%trainable_variables
&regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses* 
* 
* 
* 
c]
VARIABLE_VALUEoutput_layer/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEoutput_layer/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

+0
,1*

+0
,1*
* 

Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 

0*
'
0
1
2
3
4*

X0
Y1*
* 
* 
* 

0*
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
	
80* 
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
8
	Ztotal
	[count
\	variables
]	keras_api*
H
	^total
	_count
`
_fn_kwargs
a	variables
b	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Z0
[1*

\	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

^0
_1*

a	variables*
~
VARIABLE_VALUEAdam/relu_layer/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/relu_layer/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/output_layer/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/output_layer/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/relu_layer/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/relu_layer/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/output_layer/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/output_layer/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

*serving_default_word_embedding_layer_inputPlaceholder*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*%
shape:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¼
StatefulPartitionedCallStatefulPartitionedCall*serving_default_word_embedding_layer_inputword_embedding_layer/embeddingsrelu_layer/kernelrelu_layer/biasoutput_layer/kerneloutput_layer/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_657160
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename3word_embedding_layer/embeddings/Read/ReadVariableOp%relu_layer/kernel/Read/ReadVariableOp#relu_layer/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp,Adam/relu_layer/kernel/m/Read/ReadVariableOp*Adam/relu_layer/bias/m/Read/ReadVariableOp.Adam/output_layer/kernel/m/Read/ReadVariableOp,Adam/output_layer/bias/m/Read/ReadVariableOp,Adam/relu_layer/kernel/v/Read/ReadVariableOp*Adam/relu_layer/bias/v/Read/ReadVariableOp.Adam/output_layer/kernel/v/Read/ReadVariableOp,Adam/output_layer/bias/v/Read/ReadVariableOpConst*#
Tin
2	*
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
GPU 2J 8 *(
f#R!
__inference__traced_save_657391
Î
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameword_embedding_layer/embeddingsrelu_layer/kernelrelu_layer/biasoutput_layer/kerneloutput_layer/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/relu_layer/kernel/mAdam/relu_layer/bias/mAdam/output_layer/kernel/mAdam/output_layer/bias/mAdam/relu_layer/kernel/vAdam/relu_layer/bias/vAdam/output_layer/kernel/vAdam/output_layer/bias/v*"
Tin
2*
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_657467Ï
è$
û
F__inference_sequential_layer_call_and_return_conditional_losses_656920

inputs/
word_embedding_layer_656896:
¸×d#
relu_layer_656902:d
relu_layer_656904:%
output_layer_656908:!
output_layer_656910:
identity¢!dropout_1/StatefulPartitionedCall¢$output_layer/StatefulPartitionedCall¢"relu_layer/StatefulPartitionedCall¢3relu_layer/kernel/Regularizer/Square/ReadVariableOp¢,word_embedding_layer/StatefulPartitionedCall
,word_embedding_layer/StatefulPartitionedCallStatefulPartitionedCallinputsword_embedding_layer_656896*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_word_embedding_layer_layer_call_and_return_conditional_losses_656727d
word_embedding_layer/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
word_embedding_layer/NotEqualNotEqualinputs(word_embedding_layer/NotEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ«
(global_average_pooling1d/PartitionedCallPartitionedCall5word_embedding_layer/StatefulPartitionedCall:output:0!word_embedding_layer/NotEqual:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_656750£
"relu_layer/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0relu_layer_656902relu_layer_656904*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_relu_layer_layer_call_and_return_conditional_losses_656769ï
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall+relu_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_656849¤
$output_layer/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0output_layer_656908output_layer_656910*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_656793
3relu_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOprelu_layer_656902*
_output_shapes

:d*
dtype0
$relu_layer/kernel/Regularizer/SquareSquare;relu_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dt
#relu_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¡
!relu_layer/kernel/Regularizer/SumSum(relu_layer/kernel/Regularizer/Square:y:0,relu_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#relu_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!relu_layer/kernel/Regularizer/mulMul,relu_layer/kernel/Regularizer/mul/x:output:0*relu_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp"^dropout_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall#^relu_layer/StatefulPartitionedCall4^relu_layer/kernel/Regularizer/Square/ReadVariableOp-^word_embedding_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : 2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2H
"relu_layer/StatefulPartitionedCall"relu_layer/StatefulPartitionedCall2j
3relu_layer/kernel/Regularizer/Square/ReadVariableOp3relu_layer/kernel/Regularizer/Square/ReadVariableOp2\
,word_embedding_layer/StatefulPartitionedCall,word_embedding_layer/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
í
+__inference_sequential_layer_call_fn_657044

inputs
unknown:
¸×d
	unknown_0:d
	unknown_1:
	unknown_2:
	unknown_3:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_656920o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
:

F__inference_sequential_layer_call_and_return_conditional_losses_657090

inputs@
,word_embedding_layer_embedding_lookup_657048:
¸×d;
)relu_layer_matmul_readvariableop_resource:d8
*relu_layer_biasadd_readvariableop_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:
identity¢#output_layer/BiasAdd/ReadVariableOp¢"output_layer/MatMul/ReadVariableOp¢!relu_layer/BiasAdd/ReadVariableOp¢ relu_layer/MatMul/ReadVariableOp¢3relu_layer/kernel/Regularizer/Square/ReadVariableOp¢%word_embedding_layer/embedding_lookups
word_embedding_layer/CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
%word_embedding_layer/embedding_lookupResourceGather,word_embedding_layer_embedding_lookup_657048word_embedding_layer/Cast:y:0*
Tindices0*?
_class5
31loc:@word_embedding_layer/embedding_lookup/657048*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
dtype0ê
.word_embedding_layer/embedding_lookup/IdentityIdentity.word_embedding_layer/embedding_lookup:output:0*
T0*?
_class5
31loc:@word_embedding_layer/embedding_lookup/657048*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd´
0word_embedding_layer/embedding_lookup/Identity_1Identity7word_embedding_layer/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdd
word_embedding_layer/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
word_embedding_layer/NotEqualNotEqualinputs(word_embedding_layer/NotEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿv
,global_average_pooling1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.global_average_pooling1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.global_average_pooling1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
&global_average_pooling1d/strided_sliceStridedSlice9word_embedding_layer/embedding_lookup/Identity_1:output:05global_average_pooling1d/strided_slice/stack:output:07global_average_pooling1d/strided_slice/stack_1:output:07global_average_pooling1d/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask
global_average_pooling1d/CastCast!word_embedding_layer/NotEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿi
'global_average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Å
#global_average_pooling1d/ExpandDims
ExpandDims!global_average_pooling1d/Cast:y:00global_average_pooling1d/ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿË
global_average_pooling1d/mulMul9word_embedding_layer/embedding_lookup/Identity_1:output:0,global_average_pooling1d/ExpandDims:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdp
.global_average_pooling1d/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :°
global_average_pooling1d/SumSum global_average_pooling1d/mul:z:07global_average_pooling1d/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
0global_average_pooling1d/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :À
global_average_pooling1d/Sum_1Sum,global_average_pooling1d/ExpandDims:output:09global_average_pooling1d/Sum_1/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
 global_average_pooling1d/truedivRealDiv%global_average_pooling1d/Sum:output:0'global_average_pooling1d/Sum_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 relu_layer/MatMul/ReadVariableOpReadVariableOp)relu_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
relu_layer/MatMulMatMul$global_average_pooling1d/truediv:z:0(relu_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!relu_layer/BiasAdd/ReadVariableOpReadVariableOp*relu_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
relu_layer/BiasAddBiasAddrelu_layer/MatMul:product:0)relu_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
relu_layer/ReluRelurelu_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout_1/IdentityIdentityrelu_layer/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
output_layer/MatMulMatMuldropout_1/Identity:output:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3relu_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)relu_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
$relu_layer/kernel/Regularizer/SquareSquare;relu_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dt
#relu_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¡
!relu_layer/kernel/Regularizer/SumSum(relu_layer/kernel/Regularizer/Square:y:0,relu_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#relu_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!relu_layer/kernel/Regularizer/mulMul,relu_layer/kernel/Regularizer/mul/x:output:0*relu_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentityoutput_layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
NoOpNoOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp"^relu_layer/BiasAdd/ReadVariableOp!^relu_layer/MatMul/ReadVariableOp4^relu_layer/kernel/Regularizer/Square/ReadVariableOp&^word_embedding_layer/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : 2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2F
!relu_layer/BiasAdd/ReadVariableOp!relu_layer/BiasAdd/ReadVariableOp2D
 relu_layer/MatMul/ReadVariableOp relu_layer/MatMul/ReadVariableOp2j
3relu_layer/kernel/Regularizer/Square/ReadVariableOp3relu_layer/kernel/Regularizer/Square/ReadVariableOp2N
%word_embedding_layer/embedding_lookup%word_embedding_layer/embedding_lookup:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À#
×
F__inference_sequential_layer_call_and_return_conditional_losses_656806

inputs/
word_embedding_layer_656728:
¸×d#
relu_layer_656770:d
relu_layer_656772:%
output_layer_656794:!
output_layer_656796:
identity¢$output_layer/StatefulPartitionedCall¢"relu_layer/StatefulPartitionedCall¢3relu_layer/kernel/Regularizer/Square/ReadVariableOp¢,word_embedding_layer/StatefulPartitionedCall
,word_embedding_layer/StatefulPartitionedCallStatefulPartitionedCallinputsword_embedding_layer_656728*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_word_embedding_layer_layer_call_and_return_conditional_losses_656727d
word_embedding_layer/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
word_embedding_layer/NotEqualNotEqualinputs(word_embedding_layer/NotEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ«
(global_average_pooling1d/PartitionedCallPartitionedCall5word_embedding_layer/StatefulPartitionedCall:output:0!word_embedding_layer/NotEqual:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_656750£
"relu_layer/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0relu_layer_656770relu_layer_656772*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_relu_layer_layer_call_and_return_conditional_losses_656769ß
dropout_1/PartitionedCallPartitionedCall+relu_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_656780
$output_layer/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0output_layer_656794output_layer_656796*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_656793
3relu_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOprelu_layer_656770*
_output_shapes

:d*
dtype0
$relu_layer/kernel/Regularizer/SquareSquare;relu_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dt
#relu_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¡
!relu_layer/kernel/Regularizer/SumSum(relu_layer/kernel/Regularizer/Square:y:0,relu_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#relu_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!relu_layer/kernel/Regularizer/mulMul,relu_layer/kernel/Regularizer/mul/x:output:0*relu_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ÷
NoOpNoOp%^output_layer/StatefulPartitionedCall#^relu_layer/StatefulPartitionedCall4^relu_layer/kernel/Regularizer/Square/ReadVariableOp-^word_embedding_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : 2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2H
"relu_layer/StatefulPartitionedCall"relu_layer/StatefulPartitionedCall2j
3relu_layer/kernel/Regularizer/Square/ReadVariableOp3relu_layer/kernel/Regularizer/Square/ReadVariableOp2\
,word_embedding_layer/StatefulPartitionedCall,word_embedding_layer/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é3
¸	
__inference__traced_save_657391
file_prefix>
:savev2_word_embedding_layer_embeddings_read_readvariableop0
,savev2_relu_layer_kernel_read_readvariableop.
*savev2_relu_layer_bias_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop7
3savev2_adam_relu_layer_kernel_m_read_readvariableop5
1savev2_adam_relu_layer_bias_m_read_readvariableop9
5savev2_adam_output_layer_kernel_m_read_readvariableop7
3savev2_adam_output_layer_bias_m_read_readvariableop7
3savev2_adam_relu_layer_kernel_v_read_readvariableop5
1savev2_adam_relu_layer_bias_v_read_readvariableop9
5savev2_adam_output_layer_kernel_v_read_readvariableop7
3savev2_adam_output_layer_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ÷
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0* 
valueBB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B º	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0:savev2_word_embedding_layer_embeddings_read_readvariableop,savev2_relu_layer_kernel_read_readvariableop*savev2_relu_layer_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop3savev2_adam_relu_layer_kernel_m_read_readvariableop1savev2_adam_relu_layer_bias_m_read_readvariableop5savev2_adam_output_layer_kernel_m_read_readvariableop3savev2_adam_output_layer_bias_m_read_readvariableop3savev2_adam_relu_layer_kernel_v_read_readvariableop1savev2_adam_relu_layer_bias_v_read_readvariableop5savev2_adam_output_layer_kernel_v_read_readvariableop3savev2_adam_output_layer_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
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

identity_1Identity_1:output:0*
_input_shapes
: :
¸×d:d:::: : : : : : : : : :d::::d:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
¸×d:$ 

_output_shapes

:d: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:d: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:d: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
Ø
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_657259

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ

+__inference_relu_layer_layer_call_fn_657227

inputs
unknown:d
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_relu_layer_layer_call_and_return_conditional_losses_656769o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
÷9
°
!__inference__wrapped_model_656697
word_embedding_layer_inputK
7sequential_word_embedding_layer_embedding_lookup_656661:
¸×dF
4sequential_relu_layer_matmul_readvariableop_resource:dC
5sequential_relu_layer_biasadd_readvariableop_resource:H
6sequential_output_layer_matmul_readvariableop_resource:E
7sequential_output_layer_biasadd_readvariableop_resource:
identity¢.sequential/output_layer/BiasAdd/ReadVariableOp¢-sequential/output_layer/MatMul/ReadVariableOp¢,sequential/relu_layer/BiasAdd/ReadVariableOp¢+sequential/relu_layer/MatMul/ReadVariableOp¢0sequential/word_embedding_layer/embedding_lookup
$sequential/word_embedding_layer/CastCastword_embedding_layer_input*

DstT0*

SrcT0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÄ
0sequential/word_embedding_layer/embedding_lookupResourceGather7sequential_word_embedding_layer_embedding_lookup_656661(sequential/word_embedding_layer/Cast:y:0*
Tindices0*J
_class@
><loc:@sequential/word_embedding_layer/embedding_lookup/656661*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
dtype0
9sequential/word_embedding_layer/embedding_lookup/IdentityIdentity9sequential/word_embedding_layer/embedding_lookup:output:0*
T0*J
_class@
><loc:@sequential/word_embedding_layer/embedding_lookup/656661*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdÊ
;sequential/word_embedding_layer/embedding_lookup/Identity_1IdentityBsequential/word_embedding_layer/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdo
*sequential/word_embedding_layer/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    À
(sequential/word_embedding_layer/NotEqualNotEqualword_embedding_layer_input3sequential/word_embedding_layer/NotEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
7sequential/global_average_pooling1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9sequential/global_average_pooling1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9sequential/global_average_pooling1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¨
1sequential/global_average_pooling1d/strided_sliceStridedSliceDsequential/word_embedding_layer/embedding_lookup/Identity_1:output:0@sequential/global_average_pooling1d/strided_slice/stack:output:0Bsequential/global_average_pooling1d/strided_slice/stack_1:output:0Bsequential/global_average_pooling1d/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask¨
(sequential/global_average_pooling1d/CastCast,sequential/word_embedding_layer/NotEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿt
2sequential/global_average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :æ
.sequential/global_average_pooling1d/ExpandDims
ExpandDims,sequential/global_average_pooling1d/Cast:y:0;sequential/global_average_pooling1d/ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿì
'sequential/global_average_pooling1d/mulMulDsequential/word_embedding_layer/embedding_lookup/Identity_1:output:07sequential/global_average_pooling1d/ExpandDims:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd{
9sequential/global_average_pooling1d/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Ñ
'sequential/global_average_pooling1d/SumSum+sequential/global_average_pooling1d/mul:z:0Bsequential/global_average_pooling1d/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd}
;sequential/global_average_pooling1d/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :á
)sequential/global_average_pooling1d/Sum_1Sum7sequential/global_average_pooling1d/ExpandDims:output:0Dsequential/global_average_pooling1d/Sum_1/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÎ
+sequential/global_average_pooling1d/truedivRealDiv0sequential/global_average_pooling1d/Sum:output:02sequential/global_average_pooling1d/Sum_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd 
+sequential/relu_layer/MatMul/ReadVariableOpReadVariableOp4sequential_relu_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0¾
sequential/relu_layer/MatMulMatMul/sequential/global_average_pooling1d/truediv:z:03sequential/relu_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential/relu_layer/BiasAdd/ReadVariableOpReadVariableOp5sequential_relu_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¸
sequential/relu_layer/BiasAddBiasAdd&sequential/relu_layer/MatMul:product:04sequential/relu_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
sequential/relu_layer/ReluRelu&sequential/relu_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential/dropout_1/IdentityIdentity(sequential/relu_layer/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential/output_layer/MatMul/ReadVariableOpReadVariableOp6sequential_output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¹
sequential/output_layer/MatMulMatMul&sequential/dropout_1/Identity:output:05sequential/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential/output_layer/BiasAdd/ReadVariableOpReadVariableOp7sequential_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential/output_layer/BiasAddBiasAdd(sequential/output_layer/MatMul:product:06sequential/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential/output_layer/SoftmaxSoftmax(sequential/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
IdentityIdentity)sequential/output_layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
NoOpNoOp/^sequential/output_layer/BiasAdd/ReadVariableOp.^sequential/output_layer/MatMul/ReadVariableOp-^sequential/relu_layer/BiasAdd/ReadVariableOp,^sequential/relu_layer/MatMul/ReadVariableOp1^sequential/word_embedding_layer/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : 2`
.sequential/output_layer/BiasAdd/ReadVariableOp.sequential/output_layer/BiasAdd/ReadVariableOp2^
-sequential/output_layer/MatMul/ReadVariableOp-sequential/output_layer/MatMul/ReadVariableOp2\
,sequential/relu_layer/BiasAdd/ReadVariableOp,sequential/relu_layer/BiasAdd/ReadVariableOp2Z
+sequential/relu_layer/MatMul/ReadVariableOp+sequential/relu_layer/MatMul/ReadVariableOp2d
0sequential/word_embedding_layer/embedding_lookup0sequential/word_embedding_layer/embedding_lookup:l h
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
4
_user_specified_nameword_embedding_layer_input

p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_656707

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©
z
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_657212

inputs
mask

identity]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ú
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask\
CastCastmask*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :z

ExpandDims
ExpandDimsCast:y:0ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿf
mulMulinputsExpandDims:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdW
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :e
SumSummul:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :u
Sum_1SumExpandDims:output:0 Sum_1/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
truedivRealDivSum:output:0Sum_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:VR
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

_user_specified_namemask
ÆA

F__inference_sequential_layer_call_and_return_conditional_losses_657143

inputs@
,word_embedding_layer_embedding_lookup_657094:
¸×d;
)relu_layer_matmul_readvariableop_resource:d8
*relu_layer_biasadd_readvariableop_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:
identity¢#output_layer/BiasAdd/ReadVariableOp¢"output_layer/MatMul/ReadVariableOp¢!relu_layer/BiasAdd/ReadVariableOp¢ relu_layer/MatMul/ReadVariableOp¢3relu_layer/kernel/Regularizer/Square/ReadVariableOp¢%word_embedding_layer/embedding_lookups
word_embedding_layer/CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
%word_embedding_layer/embedding_lookupResourceGather,word_embedding_layer_embedding_lookup_657094word_embedding_layer/Cast:y:0*
Tindices0*?
_class5
31loc:@word_embedding_layer/embedding_lookup/657094*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
dtype0ê
.word_embedding_layer/embedding_lookup/IdentityIdentity.word_embedding_layer/embedding_lookup:output:0*
T0*?
_class5
31loc:@word_embedding_layer/embedding_lookup/657094*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd´
0word_embedding_layer/embedding_lookup/Identity_1Identity7word_embedding_layer/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdd
word_embedding_layer/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
word_embedding_layer/NotEqualNotEqualinputs(word_embedding_layer/NotEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿv
,global_average_pooling1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.global_average_pooling1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.global_average_pooling1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
&global_average_pooling1d/strided_sliceStridedSlice9word_embedding_layer/embedding_lookup/Identity_1:output:05global_average_pooling1d/strided_slice/stack:output:07global_average_pooling1d/strided_slice/stack_1:output:07global_average_pooling1d/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask
global_average_pooling1d/CastCast!word_embedding_layer/NotEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿi
'global_average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Å
#global_average_pooling1d/ExpandDims
ExpandDims!global_average_pooling1d/Cast:y:00global_average_pooling1d/ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿË
global_average_pooling1d/mulMul9word_embedding_layer/embedding_lookup/Identity_1:output:0,global_average_pooling1d/ExpandDims:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdp
.global_average_pooling1d/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :°
global_average_pooling1d/SumSum global_average_pooling1d/mul:z:07global_average_pooling1d/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
0global_average_pooling1d/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :À
global_average_pooling1d/Sum_1Sum,global_average_pooling1d/ExpandDims:output:09global_average_pooling1d/Sum_1/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
 global_average_pooling1d/truedivRealDiv%global_average_pooling1d/Sum:output:0'global_average_pooling1d/Sum_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 relu_layer/MatMul/ReadVariableOpReadVariableOp)relu_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
relu_layer/MatMulMatMul$global_average_pooling1d/truediv:z:0(relu_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!relu_layer/BiasAdd/ReadVariableOpReadVariableOp*relu_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
relu_layer/BiasAddBiasAddrelu_layer/MatMul:product:0)relu_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
relu_layer/ReluRelurelu_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_1/dropout/MulMulrelu_layer/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dropout_1/dropout/ShapeShaperelu_layer/Relu:activations:0*
T0*
_output_shapes
: 
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ä
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
output_layer/MatMulMatMuldropout_1/dropout/Mul_1:z:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3relu_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)relu_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
$relu_layer/kernel/Regularizer/SquareSquare;relu_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dt
#relu_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¡
!relu_layer/kernel/Regularizer/SumSum(relu_layer/kernel/Regularizer/Square:y:0,relu_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#relu_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!relu_layer/kernel/Regularizer/mulMul,relu_layer/kernel/Regularizer/mul/x:output:0*relu_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentityoutput_layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
NoOpNoOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp"^relu_layer/BiasAdd/ReadVariableOp!^relu_layer/MatMul/ReadVariableOp4^relu_layer/kernel/Regularizer/Square/ReadVariableOp&^word_embedding_layer/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : 2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2F
!relu_layer/BiasAdd/ReadVariableOp!relu_layer/BiasAdd/ReadVariableOp2D
 relu_layer/MatMul/ReadVariableOp relu_layer/MatMul/ReadVariableOp2j
3relu_layer/kernel/Regularizer/Square/ReadVariableOp3relu_layer/kernel/Regularizer/Square/ReadVariableOp2N
%word_embedding_layer/embedding_lookup%word_embedding_layer/embedding_lookup:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÖY

"__inference__traced_restore_657467
file_prefixD
0assignvariableop_word_embedding_layer_embeddings:
¸×d6
$assignvariableop_1_relu_layer_kernel:d0
"assignvariableop_2_relu_layer_bias:8
&assignvariableop_3_output_layer_kernel:2
$assignvariableop_4_output_layer_bias:&
assignvariableop_5_adam_iter:	 (
assignvariableop_6_adam_beta_1: (
assignvariableop_7_adam_beta_2: '
assignvariableop_8_adam_decay: /
%assignvariableop_9_adam_learning_rate: #
assignvariableop_10_total: #
assignvariableop_11_count: %
assignvariableop_12_total_1: %
assignvariableop_13_count_1: >
,assignvariableop_14_adam_relu_layer_kernel_m:d8
*assignvariableop_15_adam_relu_layer_bias_m:@
.assignvariableop_16_adam_output_layer_kernel_m::
,assignvariableop_17_adam_output_layer_bias_m:>
,assignvariableop_18_adam_relu_layer_kernel_v:d8
*assignvariableop_19_adam_relu_layer_bias_v:@
.assignvariableop_20_adam_output_layer_kernel_v::
,assignvariableop_21_adam_output_layer_bias_v:
identity_23¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ú
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0* 
valueBB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp0assignvariableop_word_embedding_layer_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp$assignvariableop_1_relu_layer_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_relu_layer_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp&assignvariableop_3_output_layer_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp$assignvariableop_4_output_layer_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp,assignvariableop_14_adam_relu_layer_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_relu_layer_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp.assignvariableop_16_adam_output_layer_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp,assignvariableop_17_adam_output_layer_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp,assignvariableop_18_adam_relu_layer_kernel_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_relu_layer_bias_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp.assignvariableop_20_adam_output_layer_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp,assignvariableop_21_adam_output_layer_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ³
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
:  
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_23Identity_23:output:0*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_21AssignVariableOp_212(
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
	

+__inference_sequential_layer_call_fn_656948
word_embedding_layer_input
unknown:
¸×d
	unknown_0:d
	unknown_1:
	unknown_2:
	unknown_3:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallword_embedding_layer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_656920o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:l h
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
4
_user_specified_nameword_embedding_layer_input
¸%

F__inference_sequential_layer_call_and_return_conditional_losses_657002
word_embedding_layer_input/
word_embedding_layer_656978:
¸×d#
relu_layer_656984:d
relu_layer_656986:%
output_layer_656990:!
output_layer_656992:
identity¢!dropout_1/StatefulPartitionedCall¢$output_layer/StatefulPartitionedCall¢"relu_layer/StatefulPartitionedCall¢3relu_layer/kernel/Regularizer/Square/ReadVariableOp¢,word_embedding_layer/StatefulPartitionedCall¢
,word_embedding_layer/StatefulPartitionedCallStatefulPartitionedCallword_embedding_layer_inputword_embedding_layer_656978*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_word_embedding_layer_layer_call_and_return_conditional_losses_656727d
word_embedding_layer/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ª
word_embedding_layer/NotEqualNotEqualword_embedding_layer_input(word_embedding_layer/NotEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ«
(global_average_pooling1d/PartitionedCallPartitionedCall5word_embedding_layer/StatefulPartitionedCall:output:0!word_embedding_layer/NotEqual:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_656750£
"relu_layer/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0relu_layer_656984relu_layer_656986*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_relu_layer_layer_call_and_return_conditional_losses_656769ï
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall+relu_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_656849¤
$output_layer/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0output_layer_656990output_layer_656992*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_656793
3relu_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOprelu_layer_656984*
_output_shapes

:d*
dtype0
$relu_layer/kernel/Regularizer/SquareSquare;relu_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dt
#relu_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¡
!relu_layer/kernel/Regularizer/SumSum(relu_layer/kernel/Regularizer/Square:y:0,relu_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#relu_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!relu_layer/kernel/Regularizer/mulMul,relu_layer/kernel/Regularizer/mul/x:output:0*relu_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp"^dropout_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall#^relu_layer/StatefulPartitionedCall4^relu_layer/kernel/Regularizer/Square/ReadVariableOp-^word_embedding_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : 2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2H
"relu_layer/StatefulPartitionedCall"relu_layer/StatefulPartitionedCall2j
3relu_layer/kernel/Regularizer/Square/ReadVariableOp3relu_layer/kernel/Regularizer/Square/ReadVariableOp2\
,word_embedding_layer/StatefulPartitionedCall,word_embedding_layer/StatefulPartitionedCall:l h
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
4
_user_specified_nameword_embedding_layer_input
Û
µ
__inference_loss_fn_0_657302N
<relu_layer_kernel_regularizer_square_readvariableop_resource:d
identity¢3relu_layer/kernel/Regularizer/Square/ReadVariableOp°
3relu_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<relu_layer_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:d*
dtype0
$relu_layer/kernel/Regularizer/SquareSquare;relu_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dt
#relu_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¡
!relu_layer/kernel/Regularizer/SumSum(relu_layer/kernel/Regularizer/Square:y:0,relu_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#relu_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!relu_layer/kernel/Regularizer/mulMul,relu_layer/kernel/Regularizer/mul/x:output:0*relu_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
IdentityIdentity%relu_layer/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: |
NoOpNoOp4^relu_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2j
3relu_layer/kernel/Regularizer/Square/ReadVariableOp3relu_layer/kernel/Regularizer/Square/ReadVariableOp
	

+__inference_sequential_layer_call_fn_656819
word_embedding_layer_input
unknown:
¸×d
	unknown_0:d
	unknown_1:
	unknown_2:
	unknown_3:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallword_embedding_layer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_656806o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:l h
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
4
_user_specified_nameword_embedding_layer_input
ê
ú
$__inference_signature_wrapper_657160
word_embedding_layer_input
unknown:
¸×d
	unknown_0:d
	unknown_1:
	unknown_2:
	unknown_3:
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallword_embedding_layer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_656697o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:l h
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
4
_user_specified_nameword_embedding_layer_input
î	
¯
P__inference_word_embedding_layer_layer_call_and_return_conditional_losses_656727

inputs+
embedding_lookup_656721:
¸×d
identity¢embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÄ
embedding_lookupResourceGatherembedding_lookup_656721Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/656721*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
dtype0«
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/656721*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
í
+__inference_sequential_layer_call_fn_657029

inputs
unknown:
¸×d
	unknown_0:d
	unknown_1:
	unknown_2:
	unknown_3:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_656806o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó	
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_657271

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó	
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_656849

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê

-__inference_output_layer_layer_call_fn_657280

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_656793o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
­
F__inference_relu_layer_layer_call_and_return_conditional_losses_657244

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3relu_layer/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3relu_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0
$relu_layer/kernel/Regularizer/SquareSquare;relu_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dt
#relu_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¡
!relu_layer/kernel/Regularizer/SumSum(relu_layer/kernel/Regularizer/Square:y:0,relu_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#relu_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!relu_layer/kernel/Regularizer/mulMul,relu_layer/kernel/Regularizer/mul/x:output:0*relu_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^relu_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3relu_layer/kernel/Regularizer/Square/ReadVariableOp3relu_layer/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
û
U
9__inference_global_average_pooling1d_layer_call_fn_657182

inputs
identityÈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_656707i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_656780

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
­
F__inference_relu_layer_layer_call_and_return_conditional_losses_656769

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3relu_layer/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3relu_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0
$relu_layer/kernel/Regularizer/SquareSquare;relu_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dt
#relu_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¡
!relu_layer/kernel/Regularizer/SumSum(relu_layer/kernel/Regularizer/Square:y:0,relu_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#relu_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!relu_layer/kernel/Regularizer/mulMul,relu_layer/kernel/Regularizer/mul/x:output:0*relu_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^relu_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3relu_layer/kernel/Regularizer/Square/ReadVariableOp3relu_layer/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
©
z
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_656750

inputs
mask

identity]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ú
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask\
CastCastmask*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :z

ExpandDims
ExpandDimsCast:y:0ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿf
mulMulinputsExpandDims:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdW
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :e
SumSummul:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :u
Sum_1SumExpandDims:output:0 Sum_1/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
truedivRealDivSum:output:0Sum_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:VR
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

_user_specified_namemask
Ü
_
9__inference_global_average_pooling1d_layer_call_fn_657188

inputs
mask

identityÆ
PartitionedCallPartitionedCallinputsmask*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_656750`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:VR
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

_user_specified_namemask
ñ
c
*__inference_dropout_1_layer_call_fn_657254

inputs
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_656849o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

F
*__inference_dropout_1_layer_call_fn_657249

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_656780`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à

5__inference_word_embedding_layer_layer_call_fn_657167

inputs
unknown:
¸×d
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_word_embedding_layer_layer_call_and_return_conditional_losses_656727|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
î	
¯
P__inference_word_embedding_layer_layer_call_and_return_conditional_losses_657177

inputs+
embedding_lookup_657171:
¸×d
identity¢embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÄ
embedding_lookupResourceGatherembedding_lookup_657171Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/657171*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
dtype0«
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/657171*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤

ù
H__inference_output_layer_layer_call_and_return_conditional_losses_656793

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
$
ë
F__inference_sequential_layer_call_and_return_conditional_losses_656975
word_embedding_layer_input/
word_embedding_layer_656951:
¸×d#
relu_layer_656957:d
relu_layer_656959:%
output_layer_656963:!
output_layer_656965:
identity¢$output_layer/StatefulPartitionedCall¢"relu_layer/StatefulPartitionedCall¢3relu_layer/kernel/Regularizer/Square/ReadVariableOp¢,word_embedding_layer/StatefulPartitionedCall¢
,word_embedding_layer/StatefulPartitionedCallStatefulPartitionedCallword_embedding_layer_inputword_embedding_layer_656951*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_word_embedding_layer_layer_call_and_return_conditional_losses_656727d
word_embedding_layer/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ª
word_embedding_layer/NotEqualNotEqualword_embedding_layer_input(word_embedding_layer/NotEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ«
(global_average_pooling1d/PartitionedCallPartitionedCall5word_embedding_layer/StatefulPartitionedCall:output:0!word_embedding_layer/NotEqual:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_656750£
"relu_layer/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0relu_layer_656957relu_layer_656959*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_relu_layer_layer_call_and_return_conditional_losses_656769ß
dropout_1/PartitionedCallPartitionedCall+relu_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_656780
$output_layer/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0output_layer_656963output_layer_656965*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_656793
3relu_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOprelu_layer_656957*
_output_shapes

:d*
dtype0
$relu_layer/kernel/Regularizer/SquareSquare;relu_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dt
#relu_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¡
!relu_layer/kernel/Regularizer/SumSum(relu_layer/kernel/Regularizer/Square:y:0,relu_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#relu_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8£
!relu_layer/kernel/Regularizer/mulMul,relu_layer/kernel/Regularizer/mul/x:output:0*relu_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ÷
NoOpNoOp%^output_layer/StatefulPartitionedCall#^relu_layer/StatefulPartitionedCall4^relu_layer/kernel/Regularizer/Square/ReadVariableOp-^word_embedding_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : 2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2H
"relu_layer/StatefulPartitionedCall"relu_layer/StatefulPartitionedCall2j
3relu_layer/kernel/Regularizer/Square/ReadVariableOp3relu_layer/kernel/Regularizer/Square/ReadVariableOp2\
,word_embedding_layer/StatefulPartitionedCall,word_embedding_layer/StatefulPartitionedCall:l h
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
4
_user_specified_nameword_embedding_layer_input

p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_657194

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤

ù
H__inference_output_layer_layer_call_and_return_conditional_losses_657291

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Þ
serving_defaultÊ
j
word_embedding_layer_inputL
,serving_default_word_embedding_layer_input:0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
output_layer0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:·s
õ
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
µ

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(_random_generator
)__call__
**&call_and_return_all_conditional_losses"
_tf_keras_layer
»

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer

3iter

4beta_1

5beta_2
	6decay
7learning_ratemcmd+me,mfvgvh+vi,vj"
	optimizer
C
0
1
2
+3
,4"
trackable_list_wrapper
<
0
1
+2
,3"
trackable_list_wrapper
'
80"
trackable_list_wrapper
Ê
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ú2÷
+__inference_sequential_layer_call_fn_656819
+__inference_sequential_layer_call_fn_657029
+__inference_sequential_layer_call_fn_657044
+__inference_sequential_layer_call_fn_656948À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
F__inference_sequential_layer_call_and_return_conditional_losses_657090
F__inference_sequential_layer_call_and_return_conditional_losses_657143
F__inference_sequential_layer_call_and_return_conditional_losses_656975
F__inference_sequential_layer_call_and_return_conditional_losses_657002À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ßBÜ
!__inference__wrapped_model_656697word_embedding_layer_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
>serving_default"
signature_map
3:1
¸×d2word_embedding_layer/embeddings
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ß2Ü
5__inference_word_embedding_layer_layer_call_fn_657167¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_word_embedding_layer_layer_call_and_return_conditional_losses_657177¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
«2¨
9__inference_global_average_pooling1d_layer_call_fn_657182
9__inference_global_average_pooling1d_layer_call_fn_657188¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
á2Þ
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_657194
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_657212¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
#:!d2relu_layer/kernel
:2relu_layer/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
80"
trackable_list_wrapper
­
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_relu_layer_layer_call_fn_657227¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_relu_layer_layer_call_and_return_conditional_losses_657244¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
$	variables
%trainable_variables
&regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_dropout_1_layer_call_fn_657249
*__inference_dropout_1_layer_call_fn_657254´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout_1_layer_call_and_return_conditional_losses_657259
E__inference_dropout_1_layer_call_and_return_conditional_losses_657271´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
%:#2output_layer/kernel
:2output_layer/bias
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_output_layer_layer_call_fn_657280¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_output_layer_layer_call_and_return_conditional_losses_657291¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
³2°
__inference_loss_fn_0_657302
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
'
0"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÞBÛ
$__inference_signature_wrapper_657160word_embedding_layer_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
'
0"
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
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
80"
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
trackable_dict_wrapper
N
	Ztotal
	[count
\	variables
]	keras_api"
_tf_keras_metric
^
	^total
	_count
`
_fn_kwargs
a	variables
b	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
Z0
[1"
trackable_list_wrapper
-
\	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
^0
_1"
trackable_list_wrapper
-
a	variables"
_generic_user_object
(:&d2Adam/relu_layer/kernel/m
": 2Adam/relu_layer/bias/m
*:(2Adam/output_layer/kernel/m
$:"2Adam/output_layer/bias/m
(:&d2Adam/relu_layer/kernel/v
": 2Adam/relu_layer/bias/v
*:(2Adam/output_layer/kernel/v
$:"2Adam/output_layer/bias/v¸
!__inference__wrapped_model_656697+,L¢I
B¢?
=:
word_embedding_layer_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";ª8
6
output_layer&#
output_layerÿÿÿÿÿÿÿÿÿ¥
E__inference_dropout_1_layer_call_and_return_conditional_losses_657259\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¥
E__inference_dropout_1_layer_call_and_return_conditional_losses_657271\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dropout_1_layer_call_fn_657249O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ}
*__inference_dropout_1_layer_call_fn_657254O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿÓ
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_657194{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ç
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_657212e¢b
[¢X
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
'$
maskÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 «
9__inference_global_average_pooling1d_layer_call_fn_657182nI¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¿
9__inference_global_average_pooling1d_layer_call_fn_657188e¢b
[¢X
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
'$
maskÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿd;
__inference_loss_fn_0_657302¢

¢ 
ª " ¨
H__inference_output_layer_layer_call_and_return_conditional_losses_657291\+,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_output_layer_layer_call_fn_657280O+,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_relu_layer_layer_call_and_return_conditional_losses_657244\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_relu_layer_layer_call_fn_657227O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "ÿÿÿÿÿÿÿÿÿÏ
F__inference_sequential_layer_call_and_return_conditional_losses_656975+,T¢Q
J¢G
=:
word_embedding_layer_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ï
F__inference_sequential_layer_call_and_return_conditional_losses_657002+,T¢Q
J¢G
=:
word_embedding_layer_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 º
F__inference_sequential_layer_call_and_return_conditional_losses_657090p+,@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 º
F__inference_sequential_layer_call_and_return_conditional_losses_657143p+,@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¦
+__inference_sequential_layer_call_fn_656819w+,T¢Q
J¢G
=:
word_embedding_layer_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¦
+__inference_sequential_layer_call_fn_656948w+,T¢Q
J¢G
=:
word_embedding_layer_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_657029c+,@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_657044c+,@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÙ
$__inference_signature_wrapper_657160°+,j¢g
¢ 
`ª]
[
word_embedding_layer_input=:
word_embedding_layer_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ";ª8
6
output_layer&#
output_layerÿÿÿÿÿÿÿÿÿÅ
P__inference_word_embedding_layer_layer_call_and_return_conditional_losses_657177q8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
 
5__inference_word_embedding_layer_layer_call_fn_657167d8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd