(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects a b l)
(:init 
(handempty)
(ontable a)
(ontable b)
(ontable l)
(clear a)
(clear b)
(clear l)
)
(:goal
(and
(on a b)
(on b l)
)))