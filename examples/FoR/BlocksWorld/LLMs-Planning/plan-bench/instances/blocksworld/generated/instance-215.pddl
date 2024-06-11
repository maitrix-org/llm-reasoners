(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects l c a f)
(:init 
(handempty)
(ontable l)
(ontable c)
(ontable a)
(ontable f)
(clear l)
(clear c)
(clear a)
(clear f)
)
(:goal
(and
(on l c)
(on c a)
(on a f)
)))