(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects g e i)
(:init 
(handempty)
(ontable g)
(ontable e)
(ontable i)
(clear g)
(clear e)
(clear i)
)
(:goal
(and
(on g e)
(on e i)
)))