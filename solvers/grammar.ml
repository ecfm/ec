open Core

open Utils
open Type
open Program
open FastType


type grammar = {
  logVariable: float;
  library: (program*tp*float*(tContext -> tp -> tContext*(tp list))) list;
}

let primitive_grammar primitives =
  {library = List.map primitives ~f:(fun p -> match p with
       |Primitive(t,_,_) -> (p,t, 0.0 -. (log (float_of_int (List.length primitives))),
                            compile_unifier t)
       |_ -> raise (Failure "primitive_grammar: not primitive"));
   logVariable = log 0.5
  }

let grammar_primitives g =
  {logVariable = g.logVariable;
   library = g.library |> List.filter ~f:(fun (p,_,_,_) -> is_primitive p)}

let string_of_grammar g =
  string_of_float g.logVariable ^ "\n" ^
  join ~separator:"\n" (g.library |> List.map ~f:(fun (p,t,l,_) -> Float.to_string l^"\t"^(string_of_type t)^"\t"^(string_of_program p)))

let imperative_type : (tp option) ref = ref None;;
let register_imperative_type t =
  match !imperative_type with
  | Some(_) -> assert false
  | None -> imperative_type := Some(t)

let unifying_expressions g environment request context : (program*tp list*tContext*float) list =
  (* given a grammar environment requested type and typing context,
     what are all of the possible leaves that we might use?
     These could be productions in the grammar or they could be variables. 
     Yields a sequence of:
     (leaf, argument types, context with leaf return type unified with requested type, normalized log likelihood)
  *)

  let variable_candidates =
    environment |> List.mapi ~f:(fun j t -> (Index(j),t,g.logVariable)) |>
    List.filter_map ~f:(fun (p,t,ll) ->
        let (context, t) = applyContext context t in
        let return = return_of_type t in
        if might_unify return request
        then
          try
            let context = unify context return request in
            let (context,t) = applyContext context t in            
            Some((p,arguments_of_type t,context,ll))
          with UnificationFailure -> None
        else None)
  in
  let variable_candidates = match (variable_candidates, !imperative_type) with
      | (_ :: _, Some(t)) when t = request -> 
        let terminal_indices = List.filter_map variable_candidates ~f:(fun (p,t,_,_) ->
            if t = [] then Some(get_index_value p) else None) in
        if terminal_indices = [] then variable_candidates else
          let smallest_terminal_index = fold1 min terminal_indices in
          variable_candidates |> List.filter ~f:(fun (p,t,_,_) ->
              let okay = not (is_index p) ||
                         not (t = []) ||
                         get_index_value p = smallest_terminal_index in
              (* if not okay then *)
              (*   Printf.eprintf "Pruning imperative index %s with request %s; environment=%s; smallest=%i\n" *)
              (*     (string_of_program p) *)
              (*     (string_of_type request) *)
              (*     (environment |> List.map ~f:string_of_type |> join ~separator:";") *)
              (*     smallest_terminal_index; *)
              okay)
      | _ -> variable_candidates
  in
  let nv = List.length variable_candidates |> Float.of_int |> log in
  let variable_candidates = variable_candidates |> List.map ~f:(fun (p,t,k,ll) -> (p,t,k,ll-.nv)) in

  let grammar_candidates = 
    g.library |> List.filter_map ~f:(fun (p,t,ll,u) ->
        try
          let return_type = return_of_type t in
          if not (might_unify return_type request)
          then None
          else
            let (context,arguments) = u context request in
            Some(p, arguments, context, ll)
        with UnificationFailure -> None)
  in

  let candidates = variable_candidates@grammar_candidates in
  let z = List.map ~f:(fun (_,_,_,ll) -> ll) candidates |> lse_list in
  let candidates = List.map ~f:(fun (p,t,k,ll) -> (p,t,k,ll-.z)) candidates in
  candidates

(* let likelihood_under_grammar g request expression = *)
(*   let rec walk_application_tree tree = *)
(*     match tree with *)
(*     | Apply(f,x) -> walk_application_tree f @ [x] *)
(*     | _ -> [tree] *)
(*   in *)
  
(*   let rec likelihood (r : tp) (environment: tp list) (context: tContext) (p: program) : (float*tContext) = *)
(*     match r with *)
(*     (\* a function - must start out with a sequence of lambdas *\) *)
(*     | TCon("->",[argument;return_type]) ->  *)
(*       let newEnvironment = argument :: environment in *)
(*       let body = remove_abstractions 1 p in *)
(*       likelihood return_type newEnvironment context body *)
(*     | _ -> (\* not a function - must be an application instead of a lambda *\) *)
(*       let candidates = unifying_expressions g environment r context in *)
(*       match walk_application_tree p with *)
(*       | [] -> raise (Failure "walking the application tree") *)
(*       | f::xs -> *)
(*         match List.find candidates ~f:(fun (candidate,_,_,_) -> program_equal candidate f) with *)
(*         | None -> raise (Failure ("could not find function in grammar: "^(string_of_program p))) *)
(*         | Some(_, f_t, newContext, functionLikelihood) -> *)
(*           let (f_t, newContext) = applyContext newContext f_t in *)
(*           let (argument_types, _) = arguments_and_return_of_type f_t in *)
(*           List.fold_right (List.zip_exn xs argument_types) *)
(*             ~init:(functionLikelihood,newContext) *)
(*             ~f:(fun (x,x_t) (ll,ctx) -> *)
(*                 let (x_ll,ctx) = likelihood x_t environment ctx x in *)
(*                 (ll+.x_ll,ctx)) *)
(*   in *)
  
(*   likelihood request [] empty_context expression |> fst *)

let grammar_has_recursion a g =
  g.library |> List.exists ~f:(fun (p,_,_,_) -> is_recursion_of_arity a p)


(*  *)
type contextual_grammar = {
  no_context : grammar;
  variable_context : grammar;
  contextual_library : (program * grammar list) list;
}

let show_contextual_grammar cg =
  let ls = ["No context grammar:";string_of_grammar cg.no_context;"";
            "Variable context grammar:";string_of_grammar cg.variable_context;"";
           ] @ (cg.contextual_library |> List.map ~f:(fun (e,gs) ->
      gs |> List.mapi ~f:(fun i g ->
          [Printf.sprintf "Parent %s, argument index %i:" (string_of_program e) i;
          string_of_grammar g; ""]) |> List.concat) |> List.concat)
  in join ~separator:"\n" ls

let prune_contextual_grammar (g : contextual_grammar) =
  let prune e gs =
    let t = closed_inference e in
    let argument_types = arguments_of_type t in
    List.map2_exn argument_types gs ~f:(fun argument_type g ->
        let argument_type = return_of_type argument_type in
        {logVariable=g.logVariable;
         library=g.library |> List.filter ~f:(fun (_,child_type,_,_) ->
             let child_type = return_of_type child_type in
             try
               let k, child_type = instantiate_type empty_context child_type in
               let k, argument_type = instantiate_type k argument_type in
               let _ = unify k child_type argument_type in
               true
             with UnificationFailure -> false)})
  in 
  {no_context=g.no_context;
   variable_context=g.variable_context;
   contextual_library=
     g.contextual_library |> List.map ~f:(fun (e,gs) -> (e, prune e gs))}


let make_dummy_contextual g =
  {no_context=g;
   variable_context=g;
   contextual_library =
     g.library |> List.map ~f:(fun (e,t,_,_) -> (e, arguments_of_type t |> List.map ~f:(fun _ -> g)))} |>
  prune_contextual_grammar



let deserialize_grammar g =
  let open Yojson.Basic.Util in
  let logVariable = g |> member "logVariable" |> to_float in
  let productions = g |> member "productions" |> to_list |> List.map ~f:(fun p ->
    let source = p |> member "expression" |> to_string in
    let e = parse_program source |> safe_get_some ("Error parsing: "^source) in
    let t =
      try
        infer_program_type empty_context [] e |> snd
      with UnificationFailure -> raise (Failure ("Could not type "^source))
    in
    let logProbability = p |> member "logProbability" |> to_float in
    (e,t,logProbability,compile_unifier t))
  in
  (* Successfully parsed the grammar *)
  let g = {logVariable = logVariable; library = productions;} in
  g

let deserialize_contextual_grammar j =
  let open Yojson.Basic.Util in

  {no_context = j |> member "noParent" |> deserialize_grammar;
   variable_context = j |> member "variableParent" |> deserialize_grammar;
   contextual_library =
     j |> member "productions" |> to_list |> List.map ~f:(fun production ->
         let e = production |> member "program" |> to_string |> parse_program |> get_some in
         let children = production |> member "arguments" |> to_list |> List.map ~f:deserialize_grammar in
         (e, children));} |> prune_contextual_grammar
