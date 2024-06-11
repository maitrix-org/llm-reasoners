(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects a e b k)
(:init 
(handempty)
(ontable a)
(ontable e)
(ontable b)
(ontable k)
(clear a)
(clear e)
(clear b)
(clear k)
)
(:goal
(and
(on a e)
(on e b)
(on b k)
)))