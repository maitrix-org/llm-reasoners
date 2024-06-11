(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects l d b c)
(:init 
(handempty)
(ontable l)
(ontable d)
(ontable b)
(ontable c)
(clear l)
(clear d)
(clear b)
(clear c)
)
(:goal
(and
(on l d)
(on d b)
(on b c)
)))