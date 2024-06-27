(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects b d a)
(:init 
(handempty)
(ontable b)
(ontable d)
(ontable a)
(clear b)
(clear d)
(clear a)
)
(:goal
(and
(on b d)
(on d a)
)))