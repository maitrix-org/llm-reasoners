(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects l b k i)
(:init 
(handempty)
(ontable l)
(ontable b)
(ontable k)
(ontable i)
(clear l)
(clear b)
(clear k)
(clear i)
)
(:goal
(and
(on l b)
(on b k)
(on k i)
)))