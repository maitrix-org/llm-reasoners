(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects d c k i)
(:init 
(handempty)
(ontable d)
(ontable c)
(ontable k)
(ontable i)
(clear d)
(clear c)
(clear k)
(clear i)
)
(:goal
(and
(on d c)
(on c k)
(on k i)
)))