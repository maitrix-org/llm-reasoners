(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects d l f a)
(:init 
(handempty)
(ontable d)
(ontable l)
(ontable f)
(ontable a)
(clear d)
(clear l)
(clear f)
(clear a)
)
(:goal
(and
(on d l)
(on l f)
(on f a)
)))