(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects d k b g)
(:init 
(handempty)
(ontable d)
(ontable k)
(ontable b)
(ontable g)
(clear d)
(clear k)
(clear b)
(clear g)
)
(:goal
(and
(on d k)
(on k b)
(on b g)
)))