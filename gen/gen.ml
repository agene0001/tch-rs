(* Automatically generate the C++ -> C -> rust bindings.
   This takes as input the Descriptions.yaml file that gets generated when
   building PyTorch from source.

   Run with: dune exec gen/gen.exe
*)
open Base
open Stdio

let excluded_functions =
  Set.of_list
    (module String)
    [ "multi_margin_loss"
    ; "multi_margin_loss_out"
    ; "log_softmax_backward_data"
    ; "softmax_backward_data"
    ; "clone"
    ; "copy"
    ; "copy_out"
    ; "copy_"
    ; "conv_transpose2d_backward_out"
    ; "conv_transpose3d_backward_out"
    ; "slow_conv_transpose2d_backward_out"
    ; "slow_conv_transpose3d_backward_out"
    ; "slow_conv3d_backward_out"
    ; "normal"
    ; "_cufft_set_plan_cache_max_size"
    ; "_cufft_clear_plan_cache"
    ; "backward"
    ; "_amp_non_finite_check_and_unscale_"
    ; "_cummin_helper"
    ; "_cummax_helper"
    ; "retain_grad"
    ; "_validate_sparse_coo_tensor_args"
    ; "_sparse_semi_structured_addmm"
    ; "_backward"
    ; "size"
    ; "stride"
    ; "_assert_async"
    ; "gradient"
    ; "linalg_vector_norm"
    ; "linalg_vector_norm_out"
    ; "linalg_matrix_norm"
    ; "linalg_matrix_norm_out"
    ; "linalg__powsum"
      (* Deactivate normal_out, bernoulli_out as these result in some
       ambiguous function calls. *)
    ; "normal_out"
    ; "bernoulli_out"
    ; "nested_tensor"
    ; "arange_out"
    ; "dim"
    ; "numel"
    ; "is_contiguous"
    ]

let no_tensor_options =
  Set.of_list
    (module String)
    [ "zeros_like"
    ; "empty_like"
    ; "full_like"
    ; "ones_like"
    ; "rand_like"
    ; "randint_like"
    ; "randn_like"
    ]

let prefixed_functions =
  Set.of_list
    (module String)
    [ "add"
    ; "add_"
    ; "div"
    ; "div_"
    ; "mul"
    ; "mul_"
    ; "sub"
    ; "sub_"
    ; "nll_loss"
    ; "to_mkldnn"
    ]

(* By default, scalar argument that have a default value are not available on
   the Rust side, this is to preserve the Rust api simplicity assuming that
   these scalars arguments are not often overriden.
   Adding function name [foo] in [with_optional_scalar_args] results in having
   explicit scalar arguments even if a default is present. *)
let with_optional_scalar_args = Set.of_list (module String) [ "arange"; "baddbmm" ]

let excluded_prefixes =
  [ "_thnn_"
  ; "_th_"
  ; "thnn_"
  ; "th_"
  ; "_foreach"
  ; "_amp_foreach"
  ; "_nested_tensor"
  ; "_fused_adam"
  ; "_fused_adagrad"
  ; "sym_"
  ; "_fused_sgd"
  ; "fbgemm_"
  ]

let excluded_suffixes = [ "_forward"; "_forward_out" ]
let yaml_error yaml ~msg = Printf.failwithf "%s, %s" msg (Yaml.to_string_exn yaml) ()

let extract_bool = function
  | `Bool b -> b
  | `String "true" -> true
  | `String "false" -> false
  | yaml -> yaml_error yaml ~msg:"expected bool"

let extract_list = function
  | `A l -> l
  | yaml -> yaml_error yaml ~msg:"expected list"

let extract_map = function
  | `O map -> Map.of_alist_exn (module String) map
  | yaml -> yaml_error yaml ~msg:"expected map"

let extract_string = function
  | `String s -> s
  (* The yaml spec for torch uses n which is converted to a bool. *)
  | `Bool b -> if b then "y" else "n"
  | `Float f -> Float.to_string f
  | yaml -> yaml_error yaml ~msg:"expected string"

module Func = struct
  type arg_type =
    | Bool
    | Int64
    | Int64Option
    | Double
    | DoubleOption
    | Tensor
    | TensorOption (* Tensor.t option *)
    | IntList
    | IntListOption
    | DoubleList
    | TensorOptList
    | TensorList
    | TensorOptions (* Tensor kind and device *)
    | Scalar
    | ScalarType
    | ScalarTypeOption
    | Device
    | String
    | Layout
    | LayoutOption

  type arg =
    { arg_name : string
    ; arg_type : arg_type
    ; default_value : string option
    }

  type t =
    { name : string
    ; operator_name : string
    ; overload_name : string
    ; args : arg list
    ; returns : [ `fixed of int | `dynamic | `bool | `int64_t | `double | `nothing ]
    ; (* number of tensors that are returned *)
      kind : [ `function_ | `method_ ]
    }

  let arg_type_of_string str ~is_nullable =
    match String.lowercase str with
    | "bool" -> Some Bool
    | "int64_t" -> Some (if is_nullable then Int64Option else Int64)
    | "double" -> Some (if is_nullable then DoubleOption else Double)
    | "at::tensor" -> Some (if is_nullable then TensorOption else Tensor)
    | "at::tensoroptions" -> Some TensorOptions
    | "at::intarrayref" -> Some (if is_nullable then IntListOption else IntList)
    | "at::arrayref<double>" -> Some DoubleList
    | "const c10::list<::std::optional<at::tensor>> &"
    | "const c10::list<c10::optional<at::tensor>> &" -> Some TensorOptList
    | "const at::itensorlistref &" | "at::tensorlist" -> Some TensorList
    | "at::device" -> Some Device
    | "const at::scalar &" | "at::scalar" -> Some Scalar
    | "at::scalartype" -> if is_nullable then Some ScalarTypeOption else Some ScalarType
    | "c10::string_view" -> Some String
    | "at::layout" -> Some (if is_nullable then LayoutOption else Layout)
    | _ -> None

  let c_typed_args_list t =
    List.map t.args ~f:(fun { arg_name; arg_type; _ } ->
      match arg_type with
      | IntList | IntListOption ->
        Printf.sprintf "int64_t *%s_data, int %s_len" arg_name arg_name
      | DoubleList -> Printf.sprintf "double *%s_data, int %s_len" arg_name arg_name
      | TensorOptList | TensorList ->
        Printf.sprintf "tensor *%s_data, int %s_len" arg_name arg_name
      | TensorOptions -> Printf.sprintf "int %s_kind, int %s_device" arg_name arg_name
      | String -> Printf.sprintf "char* %s_ptr, int %s_len" arg_name arg_name
      | Int64Option -> Printf.sprintf "int64_t %s_v, uint8_t %s_null" arg_name arg_name
      | DoubleOption -> Printf.sprintf "double %s_v, uint8_t %s_null" arg_name arg_name
      (* Scalars are passed by value and rebuilt inside the wrapper, saving a
         C++ heap alloc/free plus two FFI crossings per scalar argument. *)
      | Scalar ->
        Printf.sprintf "double %s_d, int64_t %s_i, int8_t %s_is_int" arg_name arg_name arg_name
      | otherwise ->
        let simple_type_cstring =
          match otherwise with
          | Bool -> "int"
          | Int64 -> "int64_t"
          | Double -> "double"
          | Tensor -> "tensor"
          | TensorOption -> "tensor"
          | ScalarType -> "int"
          | ScalarTypeOption -> "int"
          | Device -> "int"
          | Layout | LayoutOption -> "int8_t"
          | Scalar
          | Int64Option
          | DoubleOption
          | String
          | IntList
          | IntListOption
          | DoubleList
          | TensorOptList
          | TensorList
          | TensorOptions -> assert false
        in
        Printf.sprintf "%s %s" simple_type_cstring arg_name)
    |> String.concat ~sep:", "

  let c_args_list args =
    List.map args ~f:(fun { arg_name; arg_type; _ } ->
      match arg_type with
      | Tensor -> "*" ^ arg_name
      | Scalar ->
        Printf.sprintf
          "(%s_is_int ? at::Scalar((int64_t)%s_i) : at::Scalar(%s_d))"
          arg_name
          arg_name
          arg_name
      | Layout -> Printf.sprintf "static_cast<at::Layout>(%s)" arg_name
      | LayoutOption ->
        Printf.sprintf
          "(%s == -1 ? c10::nullopt : \
           c10::optional<at::Layout>(static_cast<at::Layout>(%s)))"
          arg_name
          arg_name
      | TensorOption -> Printf.sprintf "(%s ? ::std::optional<at::Tensor>(*%s) : ::std::nullopt)" arg_name arg_name
      | Bool -> "(bool)" ^ arg_name
      | IntList -> Printf.sprintf "torch::IntArrayRef(%s_data, %s_len)" arg_name arg_name
      | IntListOption ->
        Printf.sprintf
          "%s_data == nullptr ? c10::nullopt : \
           c10::optional<torch::IntArrayRef>(torch::IntArrayRef(%s_data, %s_len))"
          arg_name
          arg_name
          arg_name
      | DoubleList ->
        Printf.sprintf "at::ArrayRef<double>(%s_data, %s_len)" arg_name arg_name
      | String -> Printf.sprintf "std::string(%s_ptr, %s_len)" arg_name arg_name
      | TensorOptList ->
        Printf.sprintf "of_carray_tensor_opt(%s_data, %s_len)" arg_name arg_name
      | TensorList -> Printf.sprintf "of_carray_tensor(%s_data, %s_len)" arg_name arg_name
      | TensorOptions ->
        Printf.sprintf
          "at::device(device_of_int(%s_device)).dtype(at::ScalarType(%s_kind))"
          arg_name
          arg_name
      | Int64Option ->
        Printf.sprintf
          "%s_null ? c10::nullopt : c10::optional<int64_t>(%s_v)"
          arg_name
          arg_name
      | DoubleOption ->
        Printf.sprintf
          "%s_null ? c10::nullopt : c10::optional<double>(%s_v)"
          arg_name
          arg_name
      | ScalarType -> Printf.sprintf "at::ScalarType(%s)" arg_name
      | ScalarTypeOption ->
        Printf.sprintf
          "%s < 0 ? c10::nullopt : c10::optional<at::ScalarType>(at::ScalarType(%s))"
          arg_name
          arg_name
      | Device -> Printf.sprintf "device_of_int(%s)" arg_name
      | _ -> arg_name)
    |> String.concat ~sep:", "

  let c_call t =
    match t.kind with
    | `function_ -> Printf.sprintf "torch::%s(%s)" t.name (c_args_list t.args)
    | `method_ ->
      (match t.args with
       | head :: tail ->
         Printf.sprintf "%s->%s(%s)" head.arg_name t.name (c_args_list tail)
       | [] ->
         Printf.failwithf "Method calls should have at least one argument %s" t.name ())

  let replace_map =
    Map.of_alist_exn
      (module String)
      [ "t", "tr"
      ; "where", "where_"
      ; "view", "view_"
      ; "unsafe", "unsafe_"
      ; "to_device", "to_device_"
      ]

  let rust_name name =
    let name =
      Map.find replace_map name
      |> Option.value ~default:name
      |> String.lowercase
      |> String.substr_replace_all ~pattern:"__" ~with_:"_"
    in
    if String.is_prefix name ~prefix:"_" then "internal" ^ name else name

  let operator_name t =
    match String.lowercase t.operator_name with
    | "scatter_reduce" ->
      (* scatter_reduce is both an operator name and also obtained from the
         scatter operator when using the reduce overload. *)
      "_scatter_reduce"
    | "scatter_reduce_" -> "_scatter_reduce_"
    | other -> other

  let c_rust_args_list t =
    List.map t.args ~f:(fun arg ->
      let an = arg.arg_name in
      let single_param = Printf.sprintf "%s_: %s" an in
      match arg.arg_type with
      | Bool -> single_param "c_int"
      | Layout | LayoutOption -> single_param "i8"
      | Int64 -> single_param "i64"
      | Double -> single_param "f64"
      | Tensor -> single_param "*mut C_tensor"
      | TensorOption -> single_param "*mut C_tensor"
      | Scalar -> Printf.sprintf "%s_d: f64, %s_i: i64, %s_is_int: i8" an an an
      | ScalarType | ScalarTypeOption -> single_param "c_int"
      | Device -> single_param "c_int"
      | String -> Printf.sprintf "%s_ptr: *const u8, %s_len: c_int" an an
      | IntList | IntListOption ->
        Printf.sprintf "%s_data: *const i64, %s_len: c_int" an an
      | DoubleList -> Printf.sprintf "%s_data: *const f64, %s_len: c_int" an an
      | TensorOptList ->
        Printf.sprintf "%s_data: *const *mut C_tensor, %s_len: c_int" an an
      | TensorList -> Printf.sprintf "%s_data: *const *mut C_tensor, %s_len: c_int" an an
      | Int64Option -> Printf.sprintf "%s_v: i64, %s_null: i8" an an
      | DoubleOption -> Printf.sprintf "%s_v: f64, %s_null: i8" an an
      | TensorOptions -> Printf.sprintf "%s_kind: c_int, %s_device: c_int" an an)
    |> String.concat ~sep:", "

  let self_name = "self"
  let input_name = "input"

  let self_tensor arg =
    match arg.arg_type with
    | Tensor -> String.( = ) arg.arg_name self_name
    | _ -> false

  let input_tensor arg =
    match arg.arg_type with
    | Tensor -> String.( = ) arg.arg_name input_name
    | _ -> false

  let type_parameters t =
    let needs_scalar_parameter =
      List.exists t.args ~f:(fun arg ->
        match arg.arg_type with
        | Scalar -> true
        | _ -> false)
    in
    let needs_type_parameter =
      List.exists t.args ~f:(fun arg ->
        match arg.arg_type with
        | TensorOptList | TensorList | TensorOption -> true
        | _ -> false)
    in
    let type_parameter = if needs_type_parameter then [ "T: Borrow<Tensor>" ] else [] in
    let scalar_parameter = if needs_scalar_parameter then [ "S: Into<Scalar>" ] else [] in
    let parameters = type_parameter @ scalar_parameter in
    match parameters with
    | [] -> ""
    | p -> "<" ^ String.concat p ~sep:", " ^ ">"

  let rust_args_list t =
    match List.partition_tf t.args ~f:self_tensor with
    | [ self ], args_list -> Some self, args_list
    | _, _ ->
      (match List.partition_tf t.args ~f:input_tensor with
       | [ self ], args_list -> Some self, args_list
       | _, _ -> None, t.args)

  let rust_typed_args_list t =
    let to_string args =
      List.map args ~f:(fun arg ->
        let rust_arg_type =
          match arg.arg_type with
          | Bool -> "bool"
          | Layout -> "Layout"
          | LayoutOption -> "Option<Layout>"
          | Int64 ->
            if String.( = ) arg.arg_name "reduction" then "crate::Reduction" else "i64"
          | Double -> "f64"
          | Tensor -> "&Tensor"
          | TensorOption -> "Option<T>"
          | IntList -> "impl IntList"
          | IntListOption -> "impl IntListOption"
          | DoubleList -> "impl DoubleList"
          | TensorOptList -> "&[Option<T>]"
          | TensorList -> "&[T]"
          | String -> "&str"
          | TensorOptions -> "(Kind, Device)"
          | Int64Option -> "impl Into<Option<i64>>"
          | DoubleOption -> "impl Into<Option<f64>>"
          | Scalar -> "S"
          | ScalarType -> "Kind"
          | ScalarTypeOption -> "impl Into<Option<Kind>>"
          | Device -> "Device"
        in
        Printf.sprintf "%s: %s" (rust_name arg.arg_name) rust_arg_type)
      |> String.concat ~sep:", "
    in
    let self_arg =
      if String.is_suffix t.name ~suffix:"_" || String.( = ) t.name "set_data"
      then "&mut self"
      else "&self"
    in
    match List.partition_tf t.args ~f:self_tensor with
    | [ self ], args_list ->
      Some self.arg_name, Printf.sprintf "%s, %s" self_arg (to_string args_list)
    | _, _ ->
      (match List.partition_tf t.args ~f:input_tensor with
       | [ self ], args_list ->
         Some self.arg_name, Printf.sprintf "%s, %s" self_arg (to_string args_list)
       | _, _ -> None, to_string t.args)

  let rust_return_type t ~fallible =
    let returns =
      match t.returns with
      | `nothing -> None
      | `fixed 1 -> Some "Tensor"
      | `fixed v ->
        List.init v ~f:(fun _ -> "Tensor")
        |> String.concat ~sep:", "
        |> Printf.sprintf "(%s)"
        |> Option.some
      | `dynamic -> Some "Vec<Tensor>"
      | `bool -> Some "bool"
      | `int64_t -> Some "i64"
      | `double -> Some "f64"
    in
    match returns with
    | None -> if fallible then Printf.sprintf " -> Result<(), TchError>" else ""
    | Some returns ->
      if fallible
      then Printf.sprintf " -> Result<%s, TchError>" returns
      else Printf.sprintf " -> %s" returns

  let rust_binding_args t ~self =
    List.map t.args ~f:(fun arg ->
      let name =
        if Option.value_map self ~default:false ~f:(String.( = ) arg.arg_name)
        then "self"
        else rust_name arg.arg_name
      in
      match arg.arg_type with
      | Tensor -> Printf.sprintf "%s.c_tensor" name
      | Scalar ->
        Printf.sprintf "%s.d_value(), %s.i_value(), %s.is_int_scalar()" name name name
      | Bool -> Printf.sprintf "if %s { 1 } else { 0 }" name
      | ScalarType -> Printf.sprintf "%s.c_int()" name
      | ScalarTypeOption -> Printf.sprintf "%s.into().map_or(-1, |s| s.c_int())" name
      | Device -> Printf.sprintf "%s.c_int()" name
      | TensorOptions -> Printf.sprintf "%s.0.c_int(), %s.1.c_int()" name name
      | Int64Option -> Printf.sprintf "%s.unwrap_or(0i64), %s.is_none() as i8" name name
      | DoubleOption ->
        Printf.sprintf "%s.unwrap_or(std::f64::NAN), %s.is_none() as i8" name name
      | String -> Printf.sprintf "%s.as_ptr(), %s.len() as i32" name name
      | IntList | IntListOption | DoubleList ->
        Printf.sprintf "%s.as_ptr(), %s.len_i32()" name name
      | TensorOptList ->
        Printf.sprintf "ptr_list_opt(%s).as_ptr(), %s.len() as i32" name name
      | TensorList -> Printf.sprintf "ptr_list(%s).as_ptr(), %s.len() as i32" name name
      | TensorOption ->
        Printf.sprintf
          "%s.as_ref().map_or(std::ptr::null_mut(), |t| t.borrow().c_tensor)"
          name
      | Int64 when String.( = ) name "reduction" -> "reduction.to_int()"
      | Layout -> Printf.sprintf "%s.to_i8()" name
      | LayoutOption -> Printf.sprintf "%s.map_or(-1, |s| s.to_i8())" name
      | _ -> name)
    |> String.concat ~sep:",\n                "
end

exception Not_a_simple_arg

let read_yaml filename =
  let funcs =
    (* Split the file to avoid Yaml.of_string_exn segfaulting. *)
    In_channel.with_file filename ~f:In_channel.input_lines
    |> List.group ~break:(fun _ l -> String.length l > 0 && Char.( = ) l.[0] '-')
    |> List.concat_map ~f:(fun lines ->
         Yaml.of_string_exn (String.concat lines ~sep:"\n") |> extract_list)
  in
  printf "Read %s, got %d functions.\n%!" filename (List.length funcs);
  List.filter_map funcs ~f:(fun yaml ->
    let map = extract_map yaml in
    let name = Map.find_exn map "name" |> extract_string in
    let operator_name = Map.find_exn map "operator_name" |> extract_string in
    let overload_name = Map.find_exn map "overload_name" |> extract_string in
    let deprecated = Map.find_exn map "deprecated" |> extract_bool in
    let method_of =
      Map.find_exn map "method_of" |> extract_list |> List.map ~f:extract_string
    in
    let arguments = Map.find_exn map "arguments" |> extract_list in
    let returns =
      let is_tensor returns =
        let returns = extract_map returns in
        let return_type = Map.find_exn returns "dynamic_type" |> extract_string in
        String.( = ) return_type "at::Tensor"
      in
      let returns = Map.find_exn map "returns" |> extract_list in
      if List.is_empty returns
      then Some `nothing
      else if List.for_all returns ~f:is_tensor
      then Some (`fixed (List.length returns))
      else (
        match returns with
        | [ returns ] ->
          let return_type =
            Map.find_exn (extract_map returns) "dynamic_type" |> extract_string
          in
          (match return_type with
           | "bool" -> Some `bool
           | "int64_t" -> Some `int64_t
           | "double" -> Some `double
           | "at::TensorList" | "dynamic_type: const c10::List<c10::optional<Tensor>> &"
             -> Some `dynamic
           | _ -> None)
        | [] | _ :: _ :: _ -> None)
    in
    let kind =
      if List.exists method_of ~f:(String.( = ) "namespace")
      then Some `function_
      else if List.exists method_of ~f:(String.( = ) "Tensor")
      then Some `method_
      else None
    in
    if (not deprecated)
       && (not
             (List.exists excluded_prefixes ~f:(fun prefix ->
                String.is_prefix name ~prefix)))
       && (not
             (List.exists excluded_suffixes ~f:(fun suffix ->
                String.is_suffix name ~suffix)))
       && not (Set.mem excluded_functions name)
    then
      Option.both returns kind
      |> Option.bind ~f:(fun (returns, kind) ->
           try
             let args ~with_optional_scalar_args =
               List.filter_map arguments ~f:(fun arg ->
                 let arg = extract_map arg in
                 let arg_name = Map.find_exn arg "name" |> extract_string in
                 let arg_type = Map.find_exn arg "dynamic_type" |> extract_string in
                 let is_nullable =
                   Map.find arg "is_nullable"
                   |> Option.value_map ~default:false ~f:extract_bool
                 in
                 let default_value =
                   Map.find arg "default" |> Option.map ~f:extract_string
                 in
                 match Func.arg_type_of_string arg_type ~is_nullable with
                 | Some Scalar when Option.is_some default_value && not is_nullable ->
                   if with_optional_scalar_args
                   then Some { Func.arg_name; arg_type = Scalar; default_value }
                   else None
                 | Some TensorOptions
                   when Option.is_some default_value && Set.mem no_tensor_options name ->
                   None
                 | Some arg_type ->
                   let arg_name =
                     match arg_name, arg_type with
                     | "self", Scalar -> "self_scalar"
                     | _, _ -> arg_name
                   in
                   Some { Func.arg_name; arg_type; default_value }
                 | None ->
                   if Option.is_some default_value then None else raise Not_a_simple_arg)
             in
             let args =
               args ~with_optional_scalar_args:(Set.mem with_optional_scalar_args name)
             in
             Some [ { Func.name; operator_name; overload_name; args; returns; kind } ]
           with
           | Not_a_simple_arg -> None)
    else None)

let p out_channel s =
  Printf.ksprintf
    (fun line ->
      Out_channel.output_string out_channel line;
      Out_channel.output_char out_channel '\n')
    s

let write_cpp funcs filename =
  Out_channel.with_file (filename ^ ".cpp") ~f:(fun out_cpp ->
    Out_channel.with_file (filename ^ ".h") ~f:(fun out_h ->
      let pc s = p out_cpp s in
      let ph s = p out_h s in
      pc "// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!";
      pc "#include \"%s.h\"" (Stdlib.Filename.basename filename);
      pc "";
      ph "// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!";
      ph "#include \"torch_api.h\"";
      ph "";
      ph "extern \"C\" {";
      Map.iteri funcs ~f:(fun ~key:exported_name ~data:func ->
        let c_typed_args_list = Func.c_typed_args_list func in
        (* For zero-argument ops the out-parameter must not be followed by a
           trailing comma: `(int *out__, )` is ill-formed C/C++. *)
        let out_sep_args =
          if String.is_empty c_typed_args_list then "" else ", " ^ c_typed_args_list
        in
        match func.returns with
        (* The `nothing`/`fixed` wrappers return nullptr on success and the
           strdup'ed error message on failure so the Rust side doesn't need a
           second FFI crossing to poll the thread-local error slot. *)
        | `nothing ->
          pc "char *atg_%s(%s) {" exported_name c_typed_args_list;
          pc "  PROTECT_ERR(";
          pc "    %s;" (Func.c_call func);
          pc "  )";
          pc "}";
          pc "";
          ph "char *atg_%s(%s);" exported_name c_typed_args_list
        | `fixed ntensors ->
          pc "char *atg_%s(tensor *out__%s) {" exported_name out_sep_args;
          pc "  PROTECT_ERR(";
          pc "    auto outputs__ = %s;" (Func.c_call func);
          if ntensors = 1
          then pc "    out__[0] = new torch::Tensor(outputs__);"
          else
            for i = 0 to ntensors - 1 do
              pc "    out__[%d] = new torch::Tensor(std::get<%d>(outputs__));" i i
            done;
          pc "  )";
          pc "}";
          pc "";
          ph "char *atg_%s(tensor *%s);" exported_name out_sep_args
        | `dynamic ->
          (* Like the other return kinds, hand the (null-terminated) tensor
             array back through an out-parameter and return the error (or
             nullptr) directly, avoiding the extra error-poll FFI crossing. *)
          pc "char *atg_%s(tensor **out__%s) {" exported_name out_sep_args;
          pc "  PROTECT_ERR(";
          pc "    auto outputs__ = %s;" (Func.c_call func);
          (* the returned type is a C++ vector of tensors *)
          pc "    int sz = outputs__.size();";
          pc
            "    torch::Tensor **out = (torch::Tensor**)malloc((sz + 1) * \
             sizeof(torch::Tensor*));";
          pc "    for (int i = 0; i < sz; ++i)";
          pc "      out[i] = new torch::Tensor(outputs__[i]);";
          pc "    out[sz] = nullptr;";
          pc "    out__[0] = out;";
          pc "  )";
          pc "}";
          pc "";
          ph "char *atg_%s(tensor **out__%s);" exported_name out_sep_args
        | (`bool | `int64_t | `double) as returns ->
          (* Like the tensor-returning ops, write the result through an
             out-parameter and return the error (or nullptr) directly, so the
             Rust side does not need a second FFI crossing to poll the
             thread-local error slot after every call. *)
          let c_type =
            match returns with
            | `bool -> "int"
            | `int64_t -> "int64_t"
            | `double -> "double"
          in
          pc "char *atg_%s(%s *out__%s) {" exported_name c_type out_sep_args;
          pc "  PROTECT_ERR(";
          pc "    out__[0] = %s;" (Func.c_call func);
          pc "  )";
          pc "}";
          pc "";
          ph "char *atg_%s(%s *out__%s);" exported_name c_type out_sep_args);
      ph "}"))

let write_fallible_wrapper funcs filename =
  Out_channel.with_file filename ~f:(fun out_ml ->
    let pm s = p out_ml s in
    pm "/* THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND! */";
    pm "#![allow(clippy::all)]";
    pm "use torch_sys::*;";
    pm "use torch_sys::c_generated::*;";
    pm "use crate::{Device, Kind, Scalar, TchError, Tensor, Layout};";
    pm "use std::convert::Into;";
    pm "use std::borrow::Borrow;";
    pm "";
    pm "// Tensor-list arguments are usually short (cat/stack/index/...): a";
    pm "// fixed stack buffer avoids a heap allocation per call, falling back";
    pm "// to a Vec only for longer lists.";
    pm "enum PtrList {";
    pm "    Stack([*mut C_tensor; 16]),";
    pm "    Heap(Vec<*mut C_tensor>),";
    pm "}";
    pm "";
    pm "impl PtrList {";
    pm "    fn as_ptr(&self) -> *const *mut C_tensor {";
    pm "        match self {";
    pm "            PtrList::Stack(a) => a.as_ptr(),";
    pm "            PtrList::Heap(v) => v.as_ptr(),";
    pm "        }";
    pm "    }";
    pm "";
    pm "    fn from_iter(it: impl ExactSizeIterator<Item = *mut C_tensor>) -> PtrList {";
    pm "        if it.len() <= 16 {";
    pm "            let mut a = [std::ptr::null_mut(); 16];";
    pm "            for (i, p) in it.enumerate() {";
    pm "                a[i] = p;";
    pm "            }";
    pm "            PtrList::Stack(a)";
    pm "        } else {";
    pm "            PtrList::Heap(it.collect())";
    pm "        }";
    pm "    }";
    pm "}";
    pm "";
    pm "fn ptr_list_opt<T: Borrow<Tensor>>(l: &[Option<T>]) -> PtrList {";
    pm
      "    PtrList::from_iter(l.iter().map(|x| x.as_ref().map_or(std::ptr::null_mut(), \
       |x| x.borrow().c_tensor)))";
    pm "}";
    pm "";
    pm "fn ptr_list<T: Borrow<Tensor>>(l: &[T]) -> PtrList {";
    pm "    PtrList::from_iter(l.iter().map(|x| x.borrow().c_tensor))";
    pm "}";
    pm "";
    pm "impl Tensor {";
    Map.iteri funcs ~f:(fun ~key:exported_name ~data:(func : Func.t) ->
      let rust_name = Func.rust_name exported_name in
      let self, rust_args_list = Func.rust_typed_args_list func in
      pm "";
      pm "    pub fn f_%s%s(" rust_name (Func.type_parameters func);
      pm "        %s" rust_args_list;
      pm "    )%s {" (Func.rust_return_type func ~fallible:true);
      List.iter func.args ~f:(fun arg ->
        match arg.arg_type with
        | DoubleOption | Int64Option | Scalar ->
          let arg_name = Func.rust_name arg.arg_name in
          pm "        let %s = %s.into();" arg_name arg_name
        | _ -> ());
      match func.returns with
      | `dynamic ->
        pm "        let mut c_tensors: *mut *mut C_tensor = std::ptr::null_mut();";
        pm "        let err__ = unsafe {";
        pm "            atg_%s(&mut c_tensors," exported_name;
        pm "                %s" (Func.rust_binding_args func ~self);
        pm "            )};";
        pm "        crate::wrappers::utils::ptr_err_to_result(err__)?;";
        pm "        let mut r__ = vec![];";
        pm "        let mut i = 0;";
        pm "        loop {";
        pm "            let c__ = unsafe{*c_tensors.add(i)};";
        pm "            if c__.is_null() { break }";
        pm "            r__.push(Tensor {c_tensor: c__});";
        pm "            i += 1;";
        pm "        }";
        pm "        unsafe{libc::free(c_tensors as *mut libc::c_void)}";
        pm "        Ok(r__)";
        pm "    }"
      | `nothing ->
        pm "        let err__ = unsafe {";
        pm "            atg_%s(" exported_name;
        pm "                %s" (Func.rust_binding_args func ~self);
        pm "            )};";
        pm "        crate::wrappers::utils::ptr_err_to_result(err__)?;";
        pm "        Ok(())";
        pm "    }"
      | `fixed ntensors ->
        pm "        let mut c_tensors = [std::ptr::null_mut(); %d];" ntensors;
        pm "        let err__ = unsafe {";
        pm "            atg_%s(c_tensors.as_mut_ptr()," exported_name;
        pm "                %s" (Func.rust_binding_args func ~self);
        pm "            )};";
        pm "        crate::wrappers::utils::ptr_err_to_result(err__)?;";
        let returns =
          if ntensors = 1
          then "Tensor { c_tensor: c_tensors[0] }"
          else
            List.init ntensors ~f:(Printf.sprintf "Tensor { c_tensor: c_tensors[%d] }")
            |> String.concat ~sep:", "
            |> Printf.sprintf "(%s)"
        in
        pm "        Ok(%s)" returns;
        pm "    }"
      | (`bool | `int64_t | `double) as returns ->
        let is_bool, zero =
          match returns with
          | `bool -> true, "0i32"
          | `int64_t -> false, "0i64"
          | `double -> false, "0f64"
        in
        pm "        let mut return_ = %s;" zero;
        pm "        let err__ = unsafe {";
        pm "            atg_%s(&mut return_," exported_name;
        pm "                %s" (Func.rust_binding_args func ~self);
        pm "            )};";
        pm "        crate::wrappers::utils::ptr_err_to_result(err__)?;";
        let return_ = if is_bool then "return_ != 0" else "return_" in
        pm "        Ok(%s)" return_;
        pm "    }");
    pm "}")

let write_wrapper funcs filename =
  Out_channel.with_file filename ~f:(fun out_ml ->
    let pm s = p out_ml s in
    pm "/* THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND! */";
    pm "#![allow(clippy::all)]";
    pm "use crate::{Device, Kind, Scalar, Tensor, Layout};";
    pm "use std::convert::Into;";
    pm "use std::borrow::Borrow;";
    pm "use torch_sys::*;";
    pm "";
    pm "impl Tensor {";
    Map.iteri funcs ~f:(fun ~key:exported_name ~data:(func : Func.t) ->
      let rust_name = Func.rust_name exported_name in
      let rust_name, fallible_rust_name =
        if Set.mem prefixed_functions func.name
        then "g_" ^ rust_name, "f_" ^ rust_name
        else rust_name, "f_" ^ rust_name
      in
      pm "";
      pm "    pub fn %s%s(" rust_name (Func.type_parameters func);
      let _self, rust_args_list = Func.rust_typed_args_list func in
      pm "        %s" rust_args_list;
      pm "    )%s {" (Func.rust_return_type func ~fallible:false);
      let self, rust_args_list = Func.rust_args_list func in
      let self = if Option.is_some self then "self." else "Tensor::" in
      let rust_args_list =
        List.map rust_args_list ~f:(fun arg -> Func.rust_name arg.Func.arg_name)
        |> String.concat ~sep:", "
      in
      pm "        %s%s(%s).unwrap()" self fallible_rust_name rust_args_list;
      pm "    }");
    pm "}")

let write_ffi funcs filename =
  Out_channel.with_file filename ~f:(fun out_ml ->
    let pm s = p out_ml s in
    pm "/* THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND! */";
    pm "#[allow(clippy::all)]";
    pm "use crate::C_tensor;";
    pm "use libc::{c_char, c_int};";
    pm "";
    pm "unsafe extern \"C\" {";
    Map.iteri funcs ~f:(fun ~key:exported_name ~data:func ->
      match func.Func.returns with
      | `nothing ->
        pm
          "    pub fn atg_%s(%s) -> *mut c_char;"
          exported_name
          (Func.c_rust_args_list func)
      | `fixed _ ->
        pm
          "    pub fn atg_%s(out__: *mut *mut C_tensor, %s) -> *mut c_char;"
          exported_name
          (Func.c_rust_args_list func)
      | `dynamic ->
        pm
          "    pub fn atg_%s(out__: *mut *mut *mut C_tensor, %s) -> *mut c_char;"
          exported_name
          (Func.c_rust_args_list func)
      | (`bool | `int64_t | `double) as returns ->
        let rust_type =
          match returns with
          | `bool -> "c_int"
          | `int64_t -> "i64"
          | `double -> "f64"
        in
        pm
          "    pub fn atg_%s(out__: *mut %s, %s) -> *mut c_char;"
          exported_name
          rust_type
          (Func.c_rust_args_list func));
    pm "}")

let methods =
  let c name args =
    { Func.name
    ; operator_name = name
    ; overload_name = ""
    ; args
    ; returns = `fixed 1
    ; kind = `method_
    }
  in
  let ca arg_name arg_type = { Func.arg_name; arg_type; default_value = None } in
  [ c "grad" [ ca "self" Tensor ]
  ; c "set_requires_grad" [ ca "self" Tensor; ca "r" Bool ]
  ; c "toType" [ ca "self" Tensor; ca "scalar_type" ScalarType ]
  ; c "to" [ ca "self" Tensor; ca "device" Device ]
  ]

let run
  ~yaml_filename
  ~cpp_filename
  ~ffi_filename
  ~wrapper_filename
  ~fallible_wrapper_filename
  =
  let funcs = read_yaml yaml_filename |> List.concat in
  let funcs = methods @ funcs in
  printf "Generating code for %d functions.\n%!" (List.length funcs);
  (* Generate some unique names for overloaded functions. *)
  let funcs =
    List.map funcs ~f:(fun func -> Func.operator_name func, func)
    |> Map.of_alist_multi (module String)
    |> Map.to_alist
    |> List.concat_map ~f:(fun (name, funcs) ->
         match funcs with
         | [] -> assert false
         | [ func ] -> [ name, func ]
         | funcs ->
           let has_empty_overload =
             List.exists funcs ~f:(fun (func : Func.t) ->
               String.is_empty func.overload_name)
           in
           List.sort funcs ~compare:(fun (f1 : Func.t) (f2 : Func.t) ->
             match Int.compare (String.length f1.name) (String.length f2.name) with
             | 0 -> Int.compare (List.length f1.args) (List.length f2.args)
             | cmp -> cmp)
           |> List.mapi ~f:(fun index (func : Func.t) ->
                let operator_name = Func.operator_name func in
                let overload_name = String.lowercase func.overload_name in
                let name =
                  if String.is_empty overload_name || (index = 0 && not has_empty_overload)
                  then operator_name
                  else if String.is_suffix operator_name ~suffix:"_"
                  then operator_name ^ overload_name ^ "_"
                  else operator_name ^ "_" ^ overload_name
                in
                name, func))
    |> Map.of_alist_exn (module String)
  in
  write_cpp funcs cpp_filename;
  write_ffi funcs ffi_filename;
  write_wrapper funcs wrapper_filename;
  write_fallible_wrapper funcs fallible_wrapper_filename

let () =
  run
    ~yaml_filename:"third_party/pytorch/Declarations-v2.13.0.yaml"
    ~cpp_filename:"torch-sys/libtch/torch_api_generated"
    ~ffi_filename:"torch-sys/src/c_generated.rs"
    ~wrapper_filename:"src/wrappers/tensor_generated.rs"
    ~fallible_wrapper_filename:"src/wrappers/tensor_fallible_generated.rs"
