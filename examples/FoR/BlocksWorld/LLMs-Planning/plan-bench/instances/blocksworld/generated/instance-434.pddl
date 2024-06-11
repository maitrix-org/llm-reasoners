(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects e a b l)
(:init 
(handempty)
(ontable e)
(ontable a)
(ontable b)
(ontable l)
(clear e)
(clear a)
(clear b)
(clear l)
)
(:goal
(and
(on e a)
(on a b)
(on b l)
)))