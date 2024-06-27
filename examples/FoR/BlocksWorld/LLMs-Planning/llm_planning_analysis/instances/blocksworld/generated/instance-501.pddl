(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects h l e i)
(:init 
(handempty)
(ontable h)
(ontable l)
(ontable e)
(ontable i)
(clear h)
(clear l)
(clear e)
(clear i)
)
(:goal
(and
(on h l)
(on l e)
(on e i)
)))