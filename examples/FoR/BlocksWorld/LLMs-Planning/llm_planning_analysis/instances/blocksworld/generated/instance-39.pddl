(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects h e i)
(:init 
(handempty)
(ontable h)
(ontable e)
(ontable i)
(clear h)
(clear e)
(clear i)
)
(:goal
(and
(on h e)
(on e i)
)))