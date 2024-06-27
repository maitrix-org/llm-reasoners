(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects d i b k)
(:init 
(handempty)
(ontable d)
(ontable i)
(ontable b)
(ontable k)
(clear d)
(clear i)
(clear b)
(clear k)
)
(:goal
(and
(on d i)
(on i b)
(on b k)
)))