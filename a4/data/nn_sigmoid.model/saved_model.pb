ΑΤ
ͺϋ
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
₯
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
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
Α
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
φ
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
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68χΗ

word_embedding_layer/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ΈΧd*0
shared_name!word_embedding_layer/embeddings

3word_embedding_layer/embeddings/Read/ReadVariableOpReadVariableOpword_embedding_layer/embeddings* 
_output_shapes
:
ΈΧd*
dtype0

sigmoid_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*%
shared_namesigmoid_layer/kernel
}
(sigmoid_layer/kernel/Read/ReadVariableOpReadVariableOpsigmoid_layer/kernel*
_output_shapes

:d*
dtype0
|
sigmoid_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namesigmoid_layer/bias
u
&sigmoid_layer/bias/Read/ReadVariableOpReadVariableOpsigmoid_layer/bias*
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

Adam/sigmoid_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*,
shared_nameAdam/sigmoid_layer/kernel/m

/Adam/sigmoid_layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/sigmoid_layer/kernel/m*
_output_shapes

:d*
dtype0

Adam/sigmoid_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/sigmoid_layer/bias/m

-Adam/sigmoid_layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/sigmoid_layer/bias/m*
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

Adam/sigmoid_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*,
shared_nameAdam/sigmoid_layer/kernel/v

/Adam/sigmoid_layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/sigmoid_layer/kernel/v*
_output_shapes

:d*
dtype0

Adam/sigmoid_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/sigmoid_layer/bias/v

-Adam/sigmoid_layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/sigmoid_layer/bias/v*
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
©0
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*δ/
valueΪ/BΧ/ BΠ/
Ϋ
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
₯
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
d^
VARIABLE_VALUEsigmoid_layer/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEsigmoid_layer/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
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

VARIABLE_VALUEAdam/sigmoid_layer/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/sigmoid_layer/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/output_layer/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/output_layer/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/sigmoid_layer/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/sigmoid_layer/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/output_layer/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/output_layer/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

*serving_default_word_embedding_layer_inputPlaceholder*0
_output_shapes
:??????????????????*
dtype0*%
shape:??????????????????
Β
StatefulPartitionedCallStatefulPartitionedCall*serving_default_word_embedding_layer_inputword_embedding_layer/embeddingssigmoid_layer/kernelsigmoid_layer/biasoutput_layer/kerneloutput_layer/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_658080
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename3word_embedding_layer/embeddings/Read/ReadVariableOp(sigmoid_layer/kernel/Read/ReadVariableOp&sigmoid_layer/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp/Adam/sigmoid_layer/kernel/m/Read/ReadVariableOp-Adam/sigmoid_layer/bias/m/Read/ReadVariableOp.Adam/output_layer/kernel/m/Read/ReadVariableOp,Adam/output_layer/bias/m/Read/ReadVariableOp/Adam/sigmoid_layer/kernel/v/Read/ReadVariableOp-Adam/sigmoid_layer/bias/v/Read/ReadVariableOp.Adam/output_layer/kernel/v/Read/ReadVariableOp,Adam/output_layer/bias/v/Read/ReadVariableOpConst*#
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
__inference__traced_save_658311
ΰ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameword_embedding_layer/embeddingssigmoid_layer/kernelsigmoid_layer/biasoutput_layer/kerneloutput_layer/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/sigmoid_layer/kernel/mAdam/sigmoid_layer/bias/mAdam/output_layer/kernel/mAdam/output_layer/bias/mAdam/sigmoid_layer/kernel/vAdam/sigmoid_layer/bias/vAdam/output_layer/kernel/vAdam/output_layer/bias/v*"
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
"__inference__traced_restore_658387ρΨ
ϋ
³
I__inference_sigmoid_layer_layer_call_and_return_conditional_losses_657689

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp’6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????
6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0
'sigmoid_layer/kernel/Regularizer/SquareSquare>sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dw
&sigmoid_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ͺ
$sigmoid_layer/kernel/Regularizer/SumSum+sigmoid_layer/kernel/Regularizer/Square:y:0/sigmoid_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&sigmoid_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ρ8¬
$sigmoid_layer/kernel/Regularizer/mulMul/sigmoid_layer/kernel/Regularizer/mul/x:output:0-sigmoid_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????°
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp7^sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2p
6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
	

-__inference_sequential_1_layer_call_fn_657868
word_embedding_layer_input
unknown:
ΈΧd
	unknown_0:d
	unknown_1:
	unknown_2:
	unknown_3:
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallword_embedding_layer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_657840o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:l h
0
_output_shapes
:??????????????????
4
_user_specified_nameword_embedding_layer_input
Κ

-__inference_output_layer_layer_call_fn_658200

inputs
unknown:
	unknown_0:
identity’StatefulPartitionedCallέ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_657713o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ρ
c
*__inference_dropout_1_layer_call_fn_658174

inputs
identity’StatefulPartitionedCallΐ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_657769o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
κ
ϊ
$__inference_signature_wrapper_658080
word_embedding_layer_input
unknown:
ΈΧd
	unknown_0:d
	unknown_1:
	unknown_2:
	unknown_3:
identity’StatefulPartitionedCallρ
StatefulPartitionedCallStatefulPartitionedCallword_embedding_layer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_657617o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:l h
0
_output_shapes
:??????????????????
4
_user_specified_nameword_embedding_layer_input
$
ε
H__inference_sequential_1_layer_call_and_return_conditional_losses_657726

inputs/
word_embedding_layer_657648:
ΈΧd&
sigmoid_layer_657690:d"
sigmoid_layer_657692:%
output_layer_657714:!
output_layer_657716:
identity’$output_layer/StatefulPartitionedCall’%sigmoid_layer/StatefulPartitionedCall’6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp’,word_embedding_layer/StatefulPartitionedCall
,word_embedding_layer/StatefulPartitionedCallStatefulPartitionedCallinputsword_embedding_layer_657648*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_word_embedding_layer_layer_call_and_return_conditional_losses_657647d
word_embedding_layer/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
word_embedding_layer/NotEqualNotEqualinputs(word_embedding_layer/NotEqual/y:output:0*
T0*0
_output_shapes
:??????????????????―
*global_average_pooling1d_1/PartitionedCallPartitionedCall5word_embedding_layer/StatefulPartitionedCall:output:0!word_embedding_layer/NotEqual:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_657670±
%sigmoid_layer/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0sigmoid_layer_657690sigmoid_layer_657692*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sigmoid_layer_layer_call_and_return_conditional_losses_657689β
dropout_1/PartitionedCallPartitionedCall.sigmoid_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_657700
$output_layer/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0output_layer_657714output_layer_657716*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_657713
6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsigmoid_layer_657690*
_output_shapes

:d*
dtype0
'sigmoid_layer/kernel/Regularizer/SquareSquare>sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dw
&sigmoid_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ͺ
$sigmoid_layer/kernel/Regularizer/SumSum+sigmoid_layer/kernel/Regularizer/Square:y:0/sigmoid_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&sigmoid_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ρ8¬
$sigmoid_layer/kernel/Regularizer/mulMul/sigmoid_layer/kernel/Regularizer/mul/x:output:0-sigmoid_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????ύ
NoOpNoOp%^output_layer/StatefulPartitionedCall&^sigmoid_layer/StatefulPartitionedCall7^sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp-^word_embedding_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2N
%sigmoid_layer/StatefulPartitionedCall%sigmoid_layer/StatefulPartitionedCall2p
6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp2\
,word_embedding_layer/StatefulPartitionedCall,word_embedding_layer/StatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
€

ω
H__inference_output_layer_layer_call_and_return_conditional_losses_657713

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
§;

H__inference_sequential_1_layer_call_and_return_conditional_losses_658010

inputs@
,word_embedding_layer_embedding_lookup_657968:
ΈΧd>
,sigmoid_layer_matmul_readvariableop_resource:d;
-sigmoid_layer_biasadd_readvariableop_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:
identity’#output_layer/BiasAdd/ReadVariableOp’"output_layer/MatMul/ReadVariableOp’$sigmoid_layer/BiasAdd/ReadVariableOp’#sigmoid_layer/MatMul/ReadVariableOp’6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp’%word_embedding_layer/embedding_lookups
word_embedding_layer/CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:??????????????????
%word_embedding_layer/embedding_lookupResourceGather,word_embedding_layer_embedding_lookup_657968word_embedding_layer/Cast:y:0*
Tindices0*?
_class5
31loc:@word_embedding_layer/embedding_lookup/657968*4
_output_shapes"
 :??????????????????d*
dtype0κ
.word_embedding_layer/embedding_lookup/IdentityIdentity.word_embedding_layer/embedding_lookup:output:0*
T0*?
_class5
31loc:@word_embedding_layer/embedding_lookup/657968*4
_output_shapes"
 :??????????????????d΄
0word_embedding_layer/embedding_lookup/Identity_1Identity7word_embedding_layer/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????dd
word_embedding_layer/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
word_embedding_layer/NotEqualNotEqualinputs(word_embedding_layer/NotEqual/y:output:0*
T0*0
_output_shapes
:??????????????????x
.global_average_pooling1d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0global_average_pooling1d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0global_average_pooling1d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ω
(global_average_pooling1d_1/strided_sliceStridedSlice9word_embedding_layer/embedding_lookup/Identity_1:output:07global_average_pooling1d_1/strided_slice/stack:output:09global_average_pooling1d_1/strided_slice/stack_1:output:09global_average_pooling1d_1/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask
global_average_pooling1d_1/CastCast!word_embedding_layer/NotEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????????????k
)global_average_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Λ
%global_average_pooling1d_1/ExpandDims
ExpandDims#global_average_pooling1d_1/Cast:y:02global_average_pooling1d_1/ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :??????????????????Ο
global_average_pooling1d_1/mulMul9word_embedding_layer/embedding_lookup/Identity_1:output:0.global_average_pooling1d_1/ExpandDims:output:0*
T0*4
_output_shapes"
 :??????????????????dr
0global_average_pooling1d_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Ά
global_average_pooling1d_1/SumSum"global_average_pooling1d_1/mul:z:09global_average_pooling1d_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????dt
2global_average_pooling1d_1/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Ζ
 global_average_pooling1d_1/Sum_1Sum.global_average_pooling1d_1/ExpandDims:output:0;global_average_pooling1d_1/Sum_1/reduction_indices:output:0*
T0*'
_output_shapes
:?????????³
"global_average_pooling1d_1/truedivRealDiv'global_average_pooling1d_1/Sum:output:0)global_average_pooling1d_1/Sum_1:output:0*
T0*'
_output_shapes
:?????????d
#sigmoid_layer/MatMul/ReadVariableOpReadVariableOp,sigmoid_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0₯
sigmoid_layer/MatMulMatMul&global_average_pooling1d_1/truediv:z:0+sigmoid_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
$sigmoid_layer/BiasAdd/ReadVariableOpReadVariableOp-sigmoid_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
sigmoid_layer/BiasAddBiasAddsigmoid_layer/MatMul:product:0,sigmoid_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
sigmoid_layer/SigmoidSigmoidsigmoid_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????k
dropout_1/IdentityIdentitysigmoid_layer/Sigmoid:y:0*
T0*'
_output_shapes
:?????????
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
output_layer/MatMulMatMuldropout_1/Identity:output:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????£
6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp,sigmoid_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
'sigmoid_layer/kernel/Regularizer/SquareSquare>sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dw
&sigmoid_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ͺ
$sigmoid_layer/kernel/Regularizer/SumSum+sigmoid_layer/kernel/Regularizer/Square:y:0/sigmoid_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&sigmoid_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ρ8¬
$sigmoid_layer/kernel/Regularizer/mulMul/sigmoid_layer/kernel/Regularizer/mul/x:output:0-sigmoid_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentityoutput_layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????Ώ
NoOpNoOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp%^sigmoid_layer/BiasAdd/ReadVariableOp$^sigmoid_layer/MatMul/ReadVariableOp7^sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp&^word_embedding_layer/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2L
$sigmoid_layer/BiasAdd/ReadVariableOp$sigmoid_layer/BiasAdd/ReadVariableOp2J
#sigmoid_layer/MatMul/ReadVariableOp#sigmoid_layer/MatMul/ReadVariableOp2p
6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp2N
%word_embedding_layer/embedding_lookup%word_embedding_layer/embedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
ξ	
―
P__inference_word_embedding_layer_layer_call_and_return_conditional_losses_658097

inputs+
embedding_lookup_658091:
ΈΧd
identity’embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:??????????????????Δ
embedding_lookupResourceGatherembedding_lookup_658091Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/658091*4
_output_shapes"
 :??????????????????d*
dtype0«
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/658091*4
_output_shapes"
 :??????????????????d
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????d
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????dY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????????????: 2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
Ψ
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_657700

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Μ

.__inference_sigmoid_layer_layer_call_fn_658147

inputs
unknown:d
	unknown_0:
identity’StatefulPartitionedCallή
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sigmoid_layer_layer_call_and_return_conditional_losses_657689o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
	

-__inference_sequential_1_layer_call_fn_657739
word_embedding_layer_input
unknown:
ΈΧd
	unknown_0:d
	unknown_1:
	unknown_2:
	unknown_3:
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallword_embedding_layer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_657726o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:l h
0
_output_shapes
:??????????????????
4
_user_specified_nameword_embedding_layer_input
¨<
Π
!__inference__wrapped_model_657617
word_embedding_layer_inputM
9sequential_1_word_embedding_layer_embedding_lookup_657581:
ΈΧdK
9sequential_1_sigmoid_layer_matmul_readvariableop_resource:dH
:sequential_1_sigmoid_layer_biasadd_readvariableop_resource:J
8sequential_1_output_layer_matmul_readvariableop_resource:G
9sequential_1_output_layer_biasadd_readvariableop_resource:
identity’0sequential_1/output_layer/BiasAdd/ReadVariableOp’/sequential_1/output_layer/MatMul/ReadVariableOp’1sequential_1/sigmoid_layer/BiasAdd/ReadVariableOp’0sequential_1/sigmoid_layer/MatMul/ReadVariableOp’2sequential_1/word_embedding_layer/embedding_lookup
&sequential_1/word_embedding_layer/CastCastword_embedding_layer_input*

DstT0*

SrcT0*0
_output_shapes
:??????????????????Μ
2sequential_1/word_embedding_layer/embedding_lookupResourceGather9sequential_1_word_embedding_layer_embedding_lookup_657581*sequential_1/word_embedding_layer/Cast:y:0*
Tindices0*L
_classB
@>loc:@sequential_1/word_embedding_layer/embedding_lookup/657581*4
_output_shapes"
 :??????????????????d*
dtype0
;sequential_1/word_embedding_layer/embedding_lookup/IdentityIdentity;sequential_1/word_embedding_layer/embedding_lookup:output:0*
T0*L
_classB
@>loc:@sequential_1/word_embedding_layer/embedding_lookup/657581*4
_output_shapes"
 :??????????????????dΞ
=sequential_1/word_embedding_layer/embedding_lookup/Identity_1IdentityDsequential_1/word_embedding_layer/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????dq
,sequential_1/word_embedding_layer/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Δ
*sequential_1/word_embedding_layer/NotEqualNotEqualword_embedding_layer_input5sequential_1/word_embedding_layer/NotEqual/y:output:0*
T0*0
_output_shapes
:??????????????????
;sequential_1/global_average_pooling1d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=sequential_1/global_average_pooling1d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=sequential_1/global_average_pooling1d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ί
5sequential_1/global_average_pooling1d_1/strided_sliceStridedSliceFsequential_1/word_embedding_layer/embedding_lookup/Identity_1:output:0Dsequential_1/global_average_pooling1d_1/strided_slice/stack:output:0Fsequential_1/global_average_pooling1d_1/strided_slice/stack_1:output:0Fsequential_1/global_average_pooling1d_1/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask?
,sequential_1/global_average_pooling1d_1/CastCast.sequential_1/word_embedding_layer/NotEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????????????x
6sequential_1/global_average_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :ς
2sequential_1/global_average_pooling1d_1/ExpandDims
ExpandDims0sequential_1/global_average_pooling1d_1/Cast:y:0?sequential_1/global_average_pooling1d_1/ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :??????????????????φ
+sequential_1/global_average_pooling1d_1/mulMulFsequential_1/word_embedding_layer/embedding_lookup/Identity_1:output:0;sequential_1/global_average_pooling1d_1/ExpandDims:output:0*
T0*4
_output_shapes"
 :??????????????????d
=sequential_1/global_average_pooling1d_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :έ
+sequential_1/global_average_pooling1d_1/SumSum/sequential_1/global_average_pooling1d_1/mul:z:0Fsequential_1/global_average_pooling1d_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????d
?sequential_1/global_average_pooling1d_1/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :ν
-sequential_1/global_average_pooling1d_1/Sum_1Sum;sequential_1/global_average_pooling1d_1/ExpandDims:output:0Hsequential_1/global_average_pooling1d_1/Sum_1/reduction_indices:output:0*
T0*'
_output_shapes
:?????????Ϊ
/sequential_1/global_average_pooling1d_1/truedivRealDiv4sequential_1/global_average_pooling1d_1/Sum:output:06sequential_1/global_average_pooling1d_1/Sum_1:output:0*
T0*'
_output_shapes
:?????????dͺ
0sequential_1/sigmoid_layer/MatMul/ReadVariableOpReadVariableOp9sequential_1_sigmoid_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0Μ
!sequential_1/sigmoid_layer/MatMulMatMul3sequential_1/global_average_pooling1d_1/truediv:z:08sequential_1/sigmoid_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????¨
1sequential_1/sigmoid_layer/BiasAdd/ReadVariableOpReadVariableOp:sequential_1_sigmoid_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Η
"sequential_1/sigmoid_layer/BiasAddBiasAdd+sequential_1/sigmoid_layer/MatMul:product:09sequential_1/sigmoid_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
"sequential_1/sigmoid_layer/SigmoidSigmoid+sequential_1/sigmoid_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
sequential_1/dropout_1/IdentityIdentity&sequential_1/sigmoid_layer/Sigmoid:y:0*
T0*'
_output_shapes
:?????????¨
/sequential_1/output_layer/MatMul/ReadVariableOpReadVariableOp8sequential_1_output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ώ
 sequential_1/output_layer/MatMulMatMul(sequential_1/dropout_1/Identity:output:07sequential_1/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????¦
0sequential_1/output_layer/BiasAdd/ReadVariableOpReadVariableOp9sequential_1_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Δ
!sequential_1/output_layer/BiasAddBiasAdd*sequential_1/output_layer/MatMul:product:08sequential_1/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
!sequential_1/output_layer/SoftmaxSoftmax*sequential_1/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????z
IdentityIdentity+sequential_1/output_layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????Η
NoOpNoOp1^sequential_1/output_layer/BiasAdd/ReadVariableOp0^sequential_1/output_layer/MatMul/ReadVariableOp2^sequential_1/sigmoid_layer/BiasAdd/ReadVariableOp1^sequential_1/sigmoid_layer/MatMul/ReadVariableOp3^sequential_1/word_embedding_layer/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 2d
0sequential_1/output_layer/BiasAdd/ReadVariableOp0sequential_1/output_layer/BiasAdd/ReadVariableOp2b
/sequential_1/output_layer/MatMul/ReadVariableOp/sequential_1/output_layer/MatMul/ReadVariableOp2f
1sequential_1/sigmoid_layer/BiasAdd/ReadVariableOp1sequential_1/sigmoid_layer/BiasAdd/ReadVariableOp2d
0sequential_1/sigmoid_layer/MatMul/ReadVariableOp0sequential_1/sigmoid_layer/MatMul/ReadVariableOp2h
2sequential_1/word_embedding_layer/embedding_lookup2sequential_1/word_embedding_layer/embedding_lookup:l h
0
_output_shapes
:??????????????????
4
_user_specified_nameword_embedding_layer_input
&

H__inference_sequential_1_layer_call_and_return_conditional_losses_657922
word_embedding_layer_input/
word_embedding_layer_657898:
ΈΧd&
sigmoid_layer_657904:d"
sigmoid_layer_657906:%
output_layer_657910:!
output_layer_657912:
identity’!dropout_1/StatefulPartitionedCall’$output_layer/StatefulPartitionedCall’%sigmoid_layer/StatefulPartitionedCall’6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp’,word_embedding_layer/StatefulPartitionedCall’
,word_embedding_layer/StatefulPartitionedCallStatefulPartitionedCallword_embedding_layer_inputword_embedding_layer_657898*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_word_embedding_layer_layer_call_and_return_conditional_losses_657647d
word_embedding_layer/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ͺ
word_embedding_layer/NotEqualNotEqualword_embedding_layer_input(word_embedding_layer/NotEqual/y:output:0*
T0*0
_output_shapes
:??????????????????―
*global_average_pooling1d_1/PartitionedCallPartitionedCall5word_embedding_layer/StatefulPartitionedCall:output:0!word_embedding_layer/NotEqual:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_657670±
%sigmoid_layer/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0sigmoid_layer_657904sigmoid_layer_657906*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sigmoid_layer_layer_call_and_return_conditional_losses_657689ς
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall.sigmoid_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_657769€
$output_layer/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0output_layer_657910output_layer_657912*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_657713
6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsigmoid_layer_657904*
_output_shapes

:d*
dtype0
'sigmoid_layer/kernel/Regularizer/SquareSquare>sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dw
&sigmoid_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ͺ
$sigmoid_layer/kernel/Regularizer/SumSum+sigmoid_layer/kernel/Regularizer/Square:y:0/sigmoid_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&sigmoid_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ρ8¬
$sigmoid_layer/kernel/Regularizer/mulMul/sigmoid_layer/kernel/Regularizer/mul/x:output:0-sigmoid_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????‘
NoOpNoOp"^dropout_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall&^sigmoid_layer/StatefulPartitionedCall7^sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp-^word_embedding_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2N
%sigmoid_layer/StatefulPartitionedCall%sigmoid_layer/StatefulPartitionedCall2p
6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp2\
,word_embedding_layer/StatefulPartitionedCall,word_embedding_layer/StatefulPartitionedCall:l h
0
_output_shapes
:??????????????????
4
_user_specified_nameword_embedding_layer_input
4
Κ	
__inference__traced_save_658311
file_prefix>
:savev2_word_embedding_layer_embeddings_read_readvariableop3
/savev2_sigmoid_layer_kernel_read_readvariableop1
-savev2_sigmoid_layer_bias_read_readvariableop2
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
"savev2_count_1_read_readvariableop:
6savev2_adam_sigmoid_layer_kernel_m_read_readvariableop8
4savev2_adam_sigmoid_layer_bias_m_read_readvariableop9
5savev2_adam_output_layer_kernel_m_read_readvariableop7
3savev2_adam_output_layer_bias_m_read_readvariableop:
6savev2_adam_sigmoid_layer_kernel_v_read_readvariableop8
4savev2_adam_sigmoid_layer_bias_v_read_readvariableop9
5savev2_adam_output_layer_kernel_v_read_readvariableop7
3savev2_adam_output_layer_bias_v_read_readvariableop
savev2_const

identity_1’MergeV2Checkpointsw
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
: χ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0* 
valueBB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B Μ	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0:savev2_word_embedding_layer_embeddings_read_readvariableop/savev2_sigmoid_layer_kernel_read_readvariableop-savev2_sigmoid_layer_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop6savev2_adam_sigmoid_layer_kernel_m_read_readvariableop4savev2_adam_sigmoid_layer_bias_m_read_readvariableop5savev2_adam_output_layer_kernel_m_read_readvariableop3savev2_adam_output_layer_bias_m_read_readvariableop6savev2_adam_sigmoid_layer_kernel_v_read_readvariableop4savev2_adam_sigmoid_layer_bias_v_read_readvariableop5savev2_adam_output_layer_kernel_v_read_readvariableop3savev2_adam_output_layer_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
ΈΧd:d:::: : : : : : : : : :d::::d:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
ΈΧd:$ 

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
ή
ο
-__inference_sequential_1_layer_call_fn_657964

inputs
unknown:
ΈΧd
	unknown_0:d
	unknown_1:
	unknown_2:
	unknown_3:
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_657840o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
Ψ
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_658179

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ι$
ω
H__inference_sequential_1_layer_call_and_return_conditional_losses_657895
word_embedding_layer_input/
word_embedding_layer_657871:
ΈΧd&
sigmoid_layer_657877:d"
sigmoid_layer_657879:%
output_layer_657883:!
output_layer_657885:
identity’$output_layer/StatefulPartitionedCall’%sigmoid_layer/StatefulPartitionedCall’6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp’,word_embedding_layer/StatefulPartitionedCall’
,word_embedding_layer/StatefulPartitionedCallStatefulPartitionedCallword_embedding_layer_inputword_embedding_layer_657871*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_word_embedding_layer_layer_call_and_return_conditional_losses_657647d
word_embedding_layer/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ͺ
word_embedding_layer/NotEqualNotEqualword_embedding_layer_input(word_embedding_layer/NotEqual/y:output:0*
T0*0
_output_shapes
:??????????????????―
*global_average_pooling1d_1/PartitionedCallPartitionedCall5word_embedding_layer/StatefulPartitionedCall:output:0!word_embedding_layer/NotEqual:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_657670±
%sigmoid_layer/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0sigmoid_layer_657877sigmoid_layer_657879*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sigmoid_layer_layer_call_and_return_conditional_losses_657689β
dropout_1/PartitionedCallPartitionedCall.sigmoid_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_657700
$output_layer/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0output_layer_657883output_layer_657885*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_657713
6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsigmoid_layer_657877*
_output_shapes

:d*
dtype0
'sigmoid_layer/kernel/Regularizer/SquareSquare>sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dw
&sigmoid_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ͺ
$sigmoid_layer/kernel/Regularizer/SumSum+sigmoid_layer/kernel/Regularizer/Square:y:0/sigmoid_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&sigmoid_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ρ8¬
$sigmoid_layer/kernel/Regularizer/mulMul/sigmoid_layer/kernel/Regularizer/mul/x:output:0-sigmoid_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????ύ
NoOpNoOp%^output_layer/StatefulPartitionedCall&^sigmoid_layer/StatefulPartitionedCall7^sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp-^word_embedding_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2N
%sigmoid_layer/StatefulPartitionedCall%sigmoid_layer/StatefulPartitionedCall2p
6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp2\
,word_embedding_layer/StatefulPartitionedCall,word_embedding_layer/StatefulPartitionedCall:l h
0
_output_shapes
:??????????????????
4
_user_specified_nameword_embedding_layer_input

F
*__inference_dropout_1_layer_call_fn_658169

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
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_657700`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ή
ο
-__inference_sequential_1_layer_call_fn_657949

inputs
unknown:
ΈΧd
	unknown_0:d
	unknown_1:
	unknown_2:
	unknown_3:
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_657726o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
σ	
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_658191

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
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
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
:?????????o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ϊY
―
"__inference__traced_restore_658387
file_prefixD
0assignvariableop_word_embedding_layer_embeddings:
ΈΧd9
'assignvariableop_1_sigmoid_layer_kernel:d3
%assignvariableop_2_sigmoid_layer_bias:8
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
assignvariableop_13_count_1: A
/assignvariableop_14_adam_sigmoid_layer_kernel_m:d;
-assignvariableop_15_adam_sigmoid_layer_bias_m:@
.assignvariableop_16_adam_output_layer_kernel_m::
,assignvariableop_17_adam_output_layer_bias_m:A
/assignvariableop_18_adam_sigmoid_layer_kernel_v:d;
-assignvariableop_19_adam_sigmoid_layer_bias_v:@
.assignvariableop_20_adam_output_layer_kernel_v::
,assignvariableop_21_adam_output_layer_bias_v:
identity_23’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_3’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9ϊ
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
:
AssignVariableOp_1AssignVariableOp'assignvariableop_1_sigmoid_layer_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp%assignvariableop_2_sigmoid_layer_biasIdentity_2:output:0"/device:CPU:0*
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
: 
AssignVariableOp_14AssignVariableOp/assignvariableop_14_adam_sigmoid_layer_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp-assignvariableop_15_adam_sigmoid_layer_bias_mIdentity_15:output:0"/device:CPU:0*
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
: 
AssignVariableOp_18AssignVariableOp/assignvariableop_18_adam_sigmoid_layer_kernel_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp-assignvariableop_19_adam_sigmoid_layer_bias_vIdentity_19:output:0"/device:CPU:0*
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
ϋ
³
I__inference_sigmoid_layer_layer_call_and_return_conditional_losses_658164

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp’6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????
6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0
'sigmoid_layer/kernel/Regularizer/SquareSquare>sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dw
&sigmoid_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ͺ
$sigmoid_layer/kernel/Regularizer/SumSum+sigmoid_layer/kernel/Regularizer/Square:y:0/sigmoid_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&sigmoid_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ρ8¬
$sigmoid_layer/kernel/Regularizer/mulMul/sigmoid_layer/kernel/Regularizer/mul/x:output:0-sigmoid_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????°
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp7^sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2p
6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
«
|
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_658132

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
valueB:Ϊ
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask\
CastCastmask*

DstT0*

SrcT0
*0
_output_shapes
:??????????????????P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :z

ExpandDims
ExpandDimsCast:y:0ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :??????????????????f
mulMulinputsExpandDims:output:0*
T0*4
_output_shapes"
 :??????????????????dW
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :e
SumSummul:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????dY
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :u
Sum_1SumExpandDims:output:0 Sum_1/reduction_indices:output:0*
T0*'
_output_shapes
:?????????b
truedivRealDivSum:output:0Sum_1:output:0*
T0*'
_output_shapes
:?????????dS
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????????????d:??????????????????:\ X
4
_output_shapes"
 :??????????????????d
 
_user_specified_nameinputs:VR
0
_output_shapes
:??????????????????

_user_specified_namemask
Α%

H__inference_sequential_1_layer_call_and_return_conditional_losses_657840

inputs/
word_embedding_layer_657816:
ΈΧd&
sigmoid_layer_657822:d"
sigmoid_layer_657824:%
output_layer_657828:!
output_layer_657830:
identity’!dropout_1/StatefulPartitionedCall’$output_layer/StatefulPartitionedCall’%sigmoid_layer/StatefulPartitionedCall’6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp’,word_embedding_layer/StatefulPartitionedCall
,word_embedding_layer/StatefulPartitionedCallStatefulPartitionedCallinputsword_embedding_layer_657816*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_word_embedding_layer_layer_call_and_return_conditional_losses_657647d
word_embedding_layer/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
word_embedding_layer/NotEqualNotEqualinputs(word_embedding_layer/NotEqual/y:output:0*
T0*0
_output_shapes
:??????????????????―
*global_average_pooling1d_1/PartitionedCallPartitionedCall5word_embedding_layer/StatefulPartitionedCall:output:0!word_embedding_layer/NotEqual:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_657670±
%sigmoid_layer/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0sigmoid_layer_657822sigmoid_layer_657824*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sigmoid_layer_layer_call_and_return_conditional_losses_657689ς
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall.sigmoid_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_657769€
$output_layer/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0output_layer_657828output_layer_657830*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_657713
6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsigmoid_layer_657822*
_output_shapes

:d*
dtype0
'sigmoid_layer/kernel/Regularizer/SquareSquare>sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dw
&sigmoid_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ͺ
$sigmoid_layer/kernel/Regularizer/SumSum+sigmoid_layer/kernel/Regularizer/Square:y:0/sigmoid_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&sigmoid_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ρ8¬
$sigmoid_layer/kernel/Regularizer/mulMul/sigmoid_layer/kernel/Regularizer/mul/x:output:0-sigmoid_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????‘
NoOpNoOp"^dropout_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall&^sigmoid_layer/StatefulPartitionedCall7^sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp-^word_embedding_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2N
%sigmoid_layer/StatefulPartitionedCall%sigmoid_layer/StatefulPartitionedCall2p
6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp2\
,word_embedding_layer/StatefulPartitionedCall,word_embedding_layer/StatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
W
;__inference_global_average_pooling1d_1_layer_call_fn_658102

inputs
identityΚ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_657627i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
ξ	
―
P__inference_word_embedding_layer_layer_call_and_return_conditional_losses_657647

inputs+
embedding_lookup_657641:
ΈΧd
identity’embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:??????????????????Δ
embedding_lookupResourceGatherembedding_lookup_657641Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/657641*4
_output_shapes"
 :??????????????????d*
dtype0«
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/657641*4
_output_shapes"
 :??????????????????d
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????d
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????dY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????????????: 2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs

»
__inference_loss_fn_0_658222Q
?sigmoid_layer_kernel_regularizer_square_readvariableop_resource:d
identity’6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOpΆ
6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp?sigmoid_layer_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:d*
dtype0
'sigmoid_layer/kernel/Regularizer/SquareSquare>sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dw
&sigmoid_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ͺ
$sigmoid_layer/kernel/Regularizer/SumSum+sigmoid_layer/kernel/Regularizer/Square:y:0/sigmoid_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&sigmoid_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ρ8¬
$sigmoid_layer/kernel/Regularizer/mulMul/sigmoid_layer/kernel/Regularizer/mul/x:output:0-sigmoid_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentity(sigmoid_layer/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp7^sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2p
6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp
ΰ
a
;__inference_global_average_pooling1d_1_layer_call_fn_658108

inputs
mask

identityΘ
PartitionedCallPartitionedCallinputsmask*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_657670`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????????????d:??????????????????:\ X
4
_output_shapes"
 :??????????????????d
 
_user_specified_nameinputs:VR
0
_output_shapes
:??????????????????

_user_specified_namemask
«
|
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_657670

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
valueB:Ϊ
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask\
CastCastmask*

DstT0*

SrcT0
*0
_output_shapes
:??????????????????P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :z

ExpandDims
ExpandDimsCast:y:0ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :??????????????????f
mulMulinputsExpandDims:output:0*
T0*4
_output_shapes"
 :??????????????????dW
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :e
SumSummul:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????dY
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :u
Sum_1SumExpandDims:output:0 Sum_1/reduction_indices:output:0*
T0*'
_output_shapes
:?????????b
truedivRealDivSum:output:0Sum_1:output:0*
T0*'
_output_shapes
:?????????dS
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????????????d:??????????????????:\ X
4
_output_shapes"
 :??????????????????d
 
_user_specified_nameinputs:VR
0
_output_shapes
:??????????????????

_user_specified_namemask
σ	
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_657769

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
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
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
:?????????o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ΰ

5__inference_word_embedding_layer_layer_call_fn_658087

inputs
unknown:
ΈΧd
identity’StatefulPartitionedCallε
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_word_embedding_layer_layer_call_and_return_conditional_losses_657647|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
ιB

H__inference_sequential_1_layer_call_and_return_conditional_losses_658063

inputs@
,word_embedding_layer_embedding_lookup_658014:
ΈΧd>
,sigmoid_layer_matmul_readvariableop_resource:d;
-sigmoid_layer_biasadd_readvariableop_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:
identity’#output_layer/BiasAdd/ReadVariableOp’"output_layer/MatMul/ReadVariableOp’$sigmoid_layer/BiasAdd/ReadVariableOp’#sigmoid_layer/MatMul/ReadVariableOp’6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp’%word_embedding_layer/embedding_lookups
word_embedding_layer/CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:??????????????????
%word_embedding_layer/embedding_lookupResourceGather,word_embedding_layer_embedding_lookup_658014word_embedding_layer/Cast:y:0*
Tindices0*?
_class5
31loc:@word_embedding_layer/embedding_lookup/658014*4
_output_shapes"
 :??????????????????d*
dtype0κ
.word_embedding_layer/embedding_lookup/IdentityIdentity.word_embedding_layer/embedding_lookup:output:0*
T0*?
_class5
31loc:@word_embedding_layer/embedding_lookup/658014*4
_output_shapes"
 :??????????????????d΄
0word_embedding_layer/embedding_lookup/Identity_1Identity7word_embedding_layer/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????dd
word_embedding_layer/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
word_embedding_layer/NotEqualNotEqualinputs(word_embedding_layer/NotEqual/y:output:0*
T0*0
_output_shapes
:??????????????????x
.global_average_pooling1d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0global_average_pooling1d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0global_average_pooling1d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ω
(global_average_pooling1d_1/strided_sliceStridedSlice9word_embedding_layer/embedding_lookup/Identity_1:output:07global_average_pooling1d_1/strided_slice/stack:output:09global_average_pooling1d_1/strided_slice/stack_1:output:09global_average_pooling1d_1/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask
global_average_pooling1d_1/CastCast!word_embedding_layer/NotEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????????????k
)global_average_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Λ
%global_average_pooling1d_1/ExpandDims
ExpandDims#global_average_pooling1d_1/Cast:y:02global_average_pooling1d_1/ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :??????????????????Ο
global_average_pooling1d_1/mulMul9word_embedding_layer/embedding_lookup/Identity_1:output:0.global_average_pooling1d_1/ExpandDims:output:0*
T0*4
_output_shapes"
 :??????????????????dr
0global_average_pooling1d_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Ά
global_average_pooling1d_1/SumSum"global_average_pooling1d_1/mul:z:09global_average_pooling1d_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????dt
2global_average_pooling1d_1/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Ζ
 global_average_pooling1d_1/Sum_1Sum.global_average_pooling1d_1/ExpandDims:output:0;global_average_pooling1d_1/Sum_1/reduction_indices:output:0*
T0*'
_output_shapes
:?????????³
"global_average_pooling1d_1/truedivRealDiv'global_average_pooling1d_1/Sum:output:0)global_average_pooling1d_1/Sum_1:output:0*
T0*'
_output_shapes
:?????????d
#sigmoid_layer/MatMul/ReadVariableOpReadVariableOp,sigmoid_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0₯
sigmoid_layer/MatMulMatMul&global_average_pooling1d_1/truediv:z:0+sigmoid_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
$sigmoid_layer/BiasAdd/ReadVariableOpReadVariableOp-sigmoid_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
sigmoid_layer/BiasAddBiasAddsigmoid_layer/MatMul:product:0,sigmoid_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
sigmoid_layer/SigmoidSigmoidsigmoid_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_1/dropout/MulMulsigmoid_layer/Sigmoid:y:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????`
dropout_1/dropout/ShapeShapesigmoid_layer/Sigmoid:y:0*
T0*
_output_shapes
: 
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Δ
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
output_layer/MatMulMatMuldropout_1/dropout/Mul_1:z:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????£
6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp,sigmoid_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
'sigmoid_layer/kernel/Regularizer/SquareSquare>sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:dw
&sigmoid_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ͺ
$sigmoid_layer/kernel/Regularizer/SumSum+sigmoid_layer/kernel/Regularizer/Square:y:0/sigmoid_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&sigmoid_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ρ8¬
$sigmoid_layer/kernel/Regularizer/mulMul/sigmoid_layer/kernel/Regularizer/mul/x:output:0-sigmoid_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentityoutput_layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????Ώ
NoOpNoOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp%^sigmoid_layer/BiasAdd/ReadVariableOp$^sigmoid_layer/MatMul/ReadVariableOp7^sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp&^word_embedding_layer/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : : : 2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2L
$sigmoid_layer/BiasAdd/ReadVariableOp$sigmoid_layer/BiasAdd/ReadVariableOp2J
#sigmoid_layer/MatMul/ReadVariableOp#sigmoid_layer/MatMul/ReadVariableOp2p
6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp6sigmoid_layer/kernel/Regularizer/Square/ReadVariableOp2N
%word_embedding_layer/embedding_lookup%word_embedding_layer/embedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
€

ω
H__inference_output_layer_layer_call_and_return_conditional_losses_658211

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs

r
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_658114

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
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs

r
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_657627

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
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs"ΫL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ή
serving_defaultΚ
j
word_embedding_layer_inputL
,serving_default_word_embedding_layer_input:0??????????????????@
output_layer0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:t
υ
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
΅

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
₯
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
Ό
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
Κ
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
2?
-__inference_sequential_1_layer_call_fn_657739
-__inference_sequential_1_layer_call_fn_657949
-__inference_sequential_1_layer_call_fn_657964
-__inference_sequential_1_layer_call_fn_657868ΐ
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
kwonlydefaultsͺ 
annotationsͺ *
 
ξ2λ
H__inference_sequential_1_layer_call_and_return_conditional_losses_658010
H__inference_sequential_1_layer_call_and_return_conditional_losses_658063
H__inference_sequential_1_layer_call_and_return_conditional_losses_657895
H__inference_sequential_1_layer_call_and_return_conditional_losses_657922ΐ
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
kwonlydefaultsͺ 
annotationsͺ *
 
ίBά
!__inference__wrapped_model_657617word_embedding_layer_input"
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
annotationsͺ *
 
,
>serving_default"
signature_map
3:1
ΈΧd2word_embedding_layer/embeddings
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
ί2ά
5__inference_word_embedding_layer_layer_call_fn_658087’
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
annotationsͺ *
 
ϊ2χ
P__inference_word_embedding_layer_layer_call_and_return_conditional_losses_658097’
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
annotationsͺ *
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
―2¬
;__inference_global_average_pooling1d_1_layer_call_fn_658102
;__inference_global_average_pooling1d_1_layer_call_fn_658108―
¦²’
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults’

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ε2β
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_658114
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_658132―
¦²’
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults’

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
&:$d2sigmoid_layer/kernel
 :2sigmoid_layer/bias
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
Ψ2Υ
.__inference_sigmoid_layer_layer_call_fn_658147’
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
annotationsͺ *
 
σ2π
I__inference_sigmoid_layer_layer_call_and_return_conditional_losses_658164’
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
annotationsͺ *
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
*__inference_dropout_1_layer_call_fn_658169
*__inference_dropout_1_layer_call_fn_658174΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
Θ2Ε
E__inference_dropout_1_layer_call_and_return_conditional_losses_658179
E__inference_dropout_1_layer_call_and_return_conditional_losses_658191΄
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
kwonlydefaultsͺ 
annotationsͺ *
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
Χ2Τ
-__inference_output_layer_layer_call_fn_658200’
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
annotationsͺ *
 
ς2ο
H__inference_output_layer_layer_call_and_return_conditional_losses_658211’
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
annotationsͺ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
³2°
__inference_loss_fn_0_658222
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
annotationsͺ *’ 
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
ήBΫ
$__inference_signature_wrapper_658080word_embedding_layer_input"
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
annotationsͺ *
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
+:)d2Adam/sigmoid_layer/kernel/m
%:#2Adam/sigmoid_layer/bias/m
*:(2Adam/output_layer/kernel/m
$:"2Adam/output_layer/bias/m
+:)d2Adam/sigmoid_layer/kernel/v
%:#2Adam/sigmoid_layer/bias/v
*:(2Adam/output_layer/kernel/v
$:"2Adam/output_layer/bias/vΈ
!__inference__wrapped_model_657617+,L’I
B’?
=:
word_embedding_layer_input??????????????????
ͺ ";ͺ8
6
output_layer&#
output_layer?????????₯
E__inference_dropout_1_layer_call_and_return_conditional_losses_658179\3’0
)’&
 
inputs?????????
p 
ͺ "%’"

0?????????
 ₯
E__inference_dropout_1_layer_call_and_return_conditional_losses_658191\3’0
)’&
 
inputs?????????
p
ͺ "%’"

0?????????
 }
*__inference_dropout_1_layer_call_fn_658169O3’0
)’&
 
inputs?????????
p 
ͺ "?????????}
*__inference_dropout_1_layer_call_fn_658174O3’0
)’&
 
inputs?????????
p
ͺ "?????????Υ
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_658114{I’F
?’<
63
inputs'???????????????????????????

 
ͺ ".’+
$!
0??????????????????
 ι
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_658132e’b
[’X
-*
inputs??????????????????d
'$
mask??????????????????

ͺ "%’"

0?????????d
 ­
;__inference_global_average_pooling1d_1_layer_call_fn_658102nI’F
?’<
63
inputs'???????????????????????????

 
ͺ "!??????????????????Α
;__inference_global_average_pooling1d_1_layer_call_fn_658108e’b
[’X
-*
inputs??????????????????d
'$
mask??????????????????

ͺ "?????????d;
__inference_loss_fn_0_658222’

’ 
ͺ " ¨
H__inference_output_layer_layer_call_and_return_conditional_losses_658211\+,/’,
%’"
 
inputs?????????
ͺ "%’"

0?????????
 
-__inference_output_layer_layer_call_fn_658200O+,/’,
%’"
 
inputs?????????
ͺ "?????????Ρ
H__inference_sequential_1_layer_call_and_return_conditional_losses_657895+,T’Q
J’G
=:
word_embedding_layer_input??????????????????
p 

 
ͺ "%’"

0?????????
 Ρ
H__inference_sequential_1_layer_call_and_return_conditional_losses_657922+,T’Q
J’G
=:
word_embedding_layer_input??????????????????
p

 
ͺ "%’"

0?????????
 Ό
H__inference_sequential_1_layer_call_and_return_conditional_losses_658010p+,@’=
6’3
)&
inputs??????????????????
p 

 
ͺ "%’"

0?????????
 Ό
H__inference_sequential_1_layer_call_and_return_conditional_losses_658063p+,@’=
6’3
)&
inputs??????????????????
p

 
ͺ "%’"

0?????????
 ¨
-__inference_sequential_1_layer_call_fn_657739w+,T’Q
J’G
=:
word_embedding_layer_input??????????????????
p 

 
ͺ "?????????¨
-__inference_sequential_1_layer_call_fn_657868w+,T’Q
J’G
=:
word_embedding_layer_input??????????????????
p

 
ͺ "?????????
-__inference_sequential_1_layer_call_fn_657949c+,@’=
6’3
)&
inputs??????????????????
p 

 
ͺ "?????????
-__inference_sequential_1_layer_call_fn_657964c+,@’=
6’3
)&
inputs??????????????????
p

 
ͺ "?????????©
I__inference_sigmoid_layer_layer_call_and_return_conditional_losses_658164\/’,
%’"
 
inputs?????????d
ͺ "%’"

0?????????
 
.__inference_sigmoid_layer_layer_call_fn_658147O/’,
%’"
 
inputs?????????d
ͺ "?????????Ω
$__inference_signature_wrapper_658080°+,j’g
’ 
`ͺ]
[
word_embedding_layer_input=:
word_embedding_layer_input??????????????????";ͺ8
6
output_layer&#
output_layer?????????Ε
P__inference_word_embedding_layer_layer_call_and_return_conditional_losses_658097q8’5
.’+
)&
inputs??????????????????
ͺ "2’/
(%
0??????????????????d
 
5__inference_word_embedding_layer_layer_call_fn_658087d8’5
.’+
)&
inputs??????????????????
ͺ "%"??????????????????d