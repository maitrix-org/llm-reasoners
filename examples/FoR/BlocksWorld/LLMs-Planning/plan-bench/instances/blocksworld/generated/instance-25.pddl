(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects j a c)
(:init 
(handempty)
(ontable j)
(ontable a)
(ontable c)
(clear j)
(clear a)
(clear c)
)
(:goal
(and
(on j a)
(on a c)
)))