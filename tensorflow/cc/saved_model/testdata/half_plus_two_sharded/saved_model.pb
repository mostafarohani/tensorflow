Ç!
÷
9
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
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
<
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
A
Placeholder
output"dtype"
dtypetype"
shapeshape: 
l
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
c

SaveSlices
filename
tensor_names
shapes_and_slices	
data2T"
T
list(type)(0
H
ShardedFilename
basename	
shard

num_shards
filename
=
ShardedFilespec
basename

num_shards
filename
q
Variable
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve¦
T
a/initial_valueConst*
_output_shapes
: *
valueB
 *   ?*
dtype0
c
aVariable*
_output_shapes
: *
	container *
shape: *
dtype0*
shared_name 

a/AssignAssignaa/initial_value*
_output_shapes
: *
validate_shape(*
use_locking(*
T0*
_class

loc:@a
L
a/readIdentitya*
_output_shapes
: *
T0*
_class

loc:@a
T
b/initial_valueConst*
_output_shapes
: *
valueB
 *   @*
dtype0
c
bVariable*
_output_shapes
: *
	container *
shape: *
dtype0*
shared_name 

b/AssignAssignbb/initial_value*
_output_shapes
: *
validate_shape(*
use_locking(*
T0*
_class

loc:@b
L
b/readIdentityb*
_output_shapes
: *
T0*
_class

loc:@b
D
xPlaceholder*
_output_shapes
:*
shape: *
dtype0
8
MulMula/readx*
_output_shapes
:*
T0
8
yAddMulb/read*
_output_shapes
:*
T0
"
initNoOp	^a/Assign	^b/Assign
P

save/ConstConst*
_output_shapes
: *
valueB Bmodel*
dtype0
Q
save/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
\
save/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
x
save/ShardedFilenameShardedFilename
save/Constsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
c
save/save/tensor_namesConst*
_output_shapes
:*
valueBBaBb*
dtype0
f
save/save/shapes_and_slicesConst*
_output_shapes
:*
valueBB B *
dtype0
u
	save/save
SaveSlicessave/ShardedFilenamesave/save/tensor_namessave/save/shapes_and_slicesab*
T
2

save/control_dependencyIdentitysave/ShardedFilename
^save/save*
_output_shapes
: *
T0*'
_class
loc:@save/ShardedFilename
v
save/ShardedFilespecShardedFilespec
save/Constsave/num_shards^save/control_dependency*
_output_shapes
: 
e
save/RestoreV2/tensor_namesConst*
_output_shapes
:*
valueBBa*
dtype0
h
save/RestoreV2/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2

save/AssignAssignasave/RestoreV2*
_output_shapes
: *
validate_shape(*
use_locking(*
T0*
_class

loc:@a
g
save/RestoreV2_1/tensor_namesConst*
_output_shapes
:*
valueBBb*
dtype0
j
!save/RestoreV2_1/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
_output_shapes
:*
dtypes
2

save/Assign_1Assignbsave/RestoreV2_1*
_output_shapes
: *
validate_shape(*
use_locking(*
T0*
_class

loc:@b
8
save/restore_shardNoOp^save/Assign^save/Assign_1
-
save/restore_allNoOp^save/restore_shard"C
save/Const:0save/ShardedFilespec:0save/restore_all (5 @F8"O
trainable_variables86

a:0a/Assigna/read:0

b:0b/Assignb/read:0"E
	variables86

a:0a/Assigna/read:0

b:0b/Assignb/read:0*;

regression-

input
x:0
output
y:0
regression